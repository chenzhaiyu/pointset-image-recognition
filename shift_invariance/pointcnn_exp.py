from torchvision import datasets
from torchvision import transforms
import torch
import importlib
import sys
import os
import numpy as np
import glob
import random


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


dataset_dir = "./translation_samples/shifted_point_cloud/"
sys.path.append(os.path.join(os.path.abspath(os.getcwd()), '../models'))
MODEL = importlib.import_module('pointcnn_cls')
classifier = MODEL.get_model(10, normal_channel=True).cuda()
checkpoint = torch.load('/media/zhaiyu/7CF2DC06F2DBC296/Github/pointnet-image-parsing/log/cls/pointcnn_cifar'
                        '/checkpoints/best_model.pth')
classifier.load_state_dict(checkpoint['model_state_dict'])

num_total = 0
num_correct = 0

# # for the consistency score
# for filename in glob.iglob(os.path.join(dataset_dir, "*.txt")):
#
#     name_base = filename.split('/')[-1][6:]
#     name_list = [dataset_dir + 'shift' + str(i) + name_base for i in range(8)]
#
#     predicted_classes = [None, None]
#
#     for i, name in enumerate(random.sample(name_list, 2)):
#
#         label = int(name.split("_")[4])
#         points = np.loadtxt(name, delimiter=",", dtype=np.float32)
#         points[:, 3:6] = points[:, 3:6] / 255.0
#         points[:, 0:3] = pc_normalize(points[:, 0:3]).astype(np.float32)
#
#         points = torch.from_numpy(points).cuda()
#
#         batch = torch.from_numpy(np.zeros(512, dtype=np.int)).cuda()
#
#         prediction, _ = classifier(points, batch)
#
#         m = torch.nn.Softmax(dim=1)
#         output = m(prediction)
#         predicted_class = np.argmax(output.cpu().detach().numpy(), axis=1)
#         predicted_classes[i] = predicted_class
#         predicted_prob = np.max(output.cpu().detach().numpy())
#
#         print("filename: " + name)
#         # print("label: " + str(label))
#         print('prediction: class ' + str(predicted_class.item()) + ' with ' + 'probability ' + str(predicted_prob))
#
#     num_total = num_total + 1
#     num_correct = num_correct + (predicted_classes[0] == predicted_classes[1]).item()
#
#     print("consistency: " + str(num_correct / num_total) + " %\n")

# for an airplane image
while True:
    dataset_dir = "./shifted_biplane/"
    predicted_classes = [None for i in range(9)]
    probs = [None for i in range(9)]
    for i, filename in enumerate(glob.iglob(os.path.join(dataset_dir, "*biplane_s_000764.txt"))):

        points = np.loadtxt(filename, delimiter=",", dtype=np.float32)
        points[:, 3:6] = points[:, 3:6] / 255.0
        points[:, 0:3] = pc_normalize(points[:, 0:3]).astype(np.float32)

        points = torch.from_numpy(points).cuda()

        batch = torch.from_numpy(np.zeros(512, dtype=np.int)).cuda()

        prediction, _ = classifier(points, batch)

        m = torch.nn.Softmax(dim=1)
        output = m(prediction)
        predicted_class = np.argmax(output.cpu().detach().numpy(), axis=1)
        predicted_classes[i] = predicted_class
        predicted_prob = np.max(output.cpu().detach().numpy())

        # print("filename: " + filename)
        # # print("label: " + str(label))
        # print('class 0' + ' with ' + 'probability ' + str(output.cpu().detach().numpy()[0][0]))
        probs[i] = output.cpu().detach().numpy()[0][0]

    if np.all(np.array(probs) > 0.25):
        print(probs)
        break
