"""
Author: Benny
Date: Nov 2019
"""
from data_utils.ClsDataLoader import ClsDataLoader
from data_utils.PyGDataloader import DataLoader as PyGDataloader
import argparse
import numpy as np
import os
import torch
import datetime
import logging
from pathlib import Path
from tqdm import tqdm
import sys
import provider
import importlib
import shutil
from torch.utils.tensorboard import SummaryWriter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    """PARAMETERS"""
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size in training [default: 512]')
    parser.add_argument('--model', default='pointcnn_cls', help='model name [default: pointnet_cls]')
    parser.add_argument('--epoch', default=250, type=int, help='number of epoch in training [default: 250]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device [default: 0]')
    parser.add_argument('--num_point', type=int, default=256, help='Point Number [default: 256]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training [default: Adam]')
    parser.add_argument('--log_dir', type=str, default='pointcnn_mnist', help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate [default: 1e-4]')
    parser.add_argument('--normal', action='store_true', default=False,
                        help='Whether to use normal information [default: False]')
    parser.add_argument('--num_worker', default=4, type=int, help='Number of Dataloader workers [default: 4]')
    parser.add_argument('--num_class', default=10, type=int, help='Number of classes [default: 10]')
    parser.add_argument('--dataset_name', default='mnist', type=str, help='Dataset name: mnist, fashion, modelnet, '
                                                                            'cifar [default: mnist]')
    parser.add_argument('--data_dir', type=str, default='data/cls/mnist_point_cloud/', help='Data dir')
    parser.add_argument('--pointcnn_data_aug', action='store_true', default=False, help='Whether to use data augmentation for PointCNN')
    return parser.parse_args()


def test(model, loader, num_class=40):
    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    for j, data in tqdm(enumerate(loader), total=len(loader)):
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        classifier = model.eval()
        if args.model == 'pointcnn_cls':
            points = points.transpose(2, 1)
            if args.dataset_name == 'cifar':
                pos = points.reshape((-1, 6))
                # normalise rgb
                pos[:, 3:6] = pos[:, 3:6] / 255.0
            else:
                pos = points.reshape((-1, 3))
            x = np.arange(0, args.batch_size)
            batch = torch.from_numpy(np.repeat(x, args.num_point)).cuda()
            pred, _ = classifier(pos, batch)
        else:
            pred, _ = classifier(points)
        pred_choice = pred.data.max(1)[1]
        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
            class_acc[cat, 1] += 1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))
    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)
    return instance_acc, class_acc


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('cls')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    # DATA_PATH = 'data/modelnet40_normal_resampled/'
    DATA_PATH = args.data_dir

    # if args.model == 'pointcnn_cls':
    #     trainDataLoader = PyGDataloader(TRAIN_DATASET, args.batch_size, shuffle=True)
    #     testDataLoader = PyGDataloader(TEST_DATASET, args.batch_size, shuffle=False)
    # else:
    TRAIN_DATASET = ClsDataLoader(root=DATA_PATH, dataset_name=args.dataset_name, npoint=args.num_point,
                                  split='train',
                                  normal_channel=args.normal)
    TEST_DATASET = ClsDataLoader(root=DATA_PATH, dataset_name=args.dataset_name, npoint=args.num_point,
                                 split='test',
                                 normal_channel=args.normal)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True,
                                                  num_workers=args.num_worker, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=args.num_worker, drop_last=True)

    '''MODEL LOADING'''
    num_class = args.num_class
    MODEL = importlib.import_module(args.model)
    shutil.copy('./models/%s.py' % args.model, str(experiment_dir))
    shutil.copy('./models/pointnet_util.py', str(experiment_dir))

    classifier = MODEL.get_model(num_class, normal_channel=args.normal).cuda()
    criterion = MODEL.get_loss().cuda()

    # try:
    #     checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    #     start_epoch = checkpoint['epoch']
    #     classifier.load_state_dict(checkpoint['model_state_dict'])
    #     log_string('Use pretrain model')
    # except:
    #     log_string('No existing model, starting training from scratch...')
    #     start_epoch = 0

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/last_model.pth')
        start_epoch = checkpoint['epoch'] + 1
        classifier.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        best_instance_acc = checkpoint['instance_acc']
        best_class_acc = checkpoint['class_acc']
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        best_instance_acc = 0.0
        best_class_acc = 0.0

    global_epoch = 0
    global_step = 0
    mean_correct = []

    '''TRAINING'''
    logger.info('Start training...')
    writer_loss = SummaryWriter(os.path.join(str(log_dir), 'loss'))
    writer_train_instance_accuracy = SummaryWriter(os.path.join(str(log_dir), 'train_instance_accuracy'))
    writer_test_instance_accuracy = SummaryWriter(os.path.join(str(log_dir), 'test_instance_accuracy'))
    writer_test_class_accuracy = SummaryWriter(os.path.join(str(log_dir), 'test_class_accuracy'))
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))

        scheduler.step()
        log_string('lr: %f' % optimizer.param_groups[0]['lr'])
        running_loss = 0.0
        for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            points, target = data
            points = points.data.numpy()
            points = provider.random_point_dropout(points)
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            if args.model=='pointcnn_cls' and args.pointcnn_data_aug == True:
                points = provider.shuffle_points(points)
                points[:, :, 0:3] = provider.rotate_point_cloud(points[:, :, 0:3])
                points[:, :, 0:3] = provider.jitter_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            target = target[:, 0]

            points = points.transpose(2, 1)

            points, target = points.cuda(), target.cuda()
            optimizer.zero_grad()

            classifier = classifier.train()
            if args.model == 'pointcnn_cls':
                points = points.transpose(2, 1)
                if args.dataset_name == 'cifar':
                    pos = points.reshape((-1, 6))
                    # normalise rgb
                    pos[:, 3:6] = pos[:, 3:6] / 255.0
                else:
                    pos = points.reshape((-1, 3))
                x = np.arange(0, args.batch_size)
                batch = torch.from_numpy(np.repeat(x, args.num_point)).cuda()
                pred, trans_feat = classifier(pos, batch)
            else:
                pred, trans_feat = classifier(points)

            loss = criterion(pred, target.long(), trans_feat)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            loss.backward()
            optimizer.step()
            global_step += 1

            running_loss += loss.item()
            if batch_id % 10 == 9:  # print every 10 batches
                niter = epoch * len(trainDataLoader) + batch_id
                writer_loss.add_scalar('Train/loss', loss.item(), niter)

        log_string('Loss: %f' % (running_loss / len(trainDataLoader)))
        train_instance_acc = np.mean(mean_correct)
        log_string('Train Instance Accuracy: %f' % train_instance_acc)
        writer_train_instance_accuracy.add_scalar('Train/instance_accuracy', train_instance_acc.item(), epoch)

        with torch.no_grad():
            instance_acc, class_acc = test(classifier.eval(), testDataLoader, num_class)
            writer_test_instance_accuracy.add_scalar('Test/instance_accuracy', instance_acc.item(), epoch)
            writer_test_class_accuracy.add_scalar('Test/class_accuracy', class_acc.item(), epoch)

            if instance_acc >= best_instance_acc:
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

            if class_acc >= best_class_acc:
                best_class_acc = class_acc
            log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
            log_string('Best Instance Accuracy: %f, Class Accuracy: %f' % (best_instance_acc, best_class_acc))

            logger.info('Save the last model...')
            savepath_last = str(checkpoints_dir) + '/last_model.pth'
            log_string('Saving at %s' % savepath_last)
            state_last = {
                'epoch': epoch,
                'instance_acc': instance_acc,
                'class_acc': class_acc,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }
            torch.save(state_last, savepath_last)

            if instance_acc >= best_instance_acc:
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    'class_acc': class_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1

    logger.info('End of training...')
    writer_loss.close()
    writer_train_instance_accuracy.close()
    writer_test_instance_accuracy.close()
    writer_test_class_accuracy.close()


if __name__ == '__main__':
    args = parse_args()
    main(args)
