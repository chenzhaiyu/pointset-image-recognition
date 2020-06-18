from torchvision import datasets
from torchvision import transforms
import torch
import importlib
import sys
import os
import numpy as np
import glob


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

sys.path.append(os.path.join(os.path.abspath(os.getcwd()), '../models'))
MODEL = importlib.import_module('pointcnn_cls')
classifier = MODEL.get_model(10, normal_channel=True).cuda()
checkpoint = torch.load('/media/zhaiyu/7CF2DC06F2DBC296/Github/pointnet-image-parsing/log/cls/pointcnn_cifar/checkpoints/best_model.pth')
classifier.load_state_dict(checkpoint['model_state_dict'])

images = []
filenames = []

for f in glob.iglob("./point_cloud_data/*jet*"):
    images.append((np.loadtxt(f, delimiter=',', dtype=np.float32)))
    filenames.append(f)


for image, filename in zip(images, filenames):
    # normalise rgb
    image[:, 3:6] = image[:, 3:6] / 255.0
    image[:, 0:3] = pc_normalize(image[:, 0:3])

    image = torch.from_numpy(image).cuda()

    # todo: 256 vs 512
    batch = torch.from_numpy(np.zeros(512, dtype=np.int)).cuda()

    pred, _ = classifier(image, batch)

    m = torch.nn.Softmax(dim=1)
    output = m(pred)

    print(filename)
    print('prediction: class ' + str(np.argmax(output.cpu().detach().numpy(), axis=1)) + ' with ' + 'probability ' + str(np.max(output.cpu().detach().numpy())))
    print("\n")
