""" Convert MNIST and Fashion MNIST image dataset to point cloud """

import os
from tqdm import tqdm
import glob
import random
import numpy as np


def convert2txt(img_file, label_file, txt_file, n_images):
    """ Convert the original MNIST data to text files """

    f = open(img_file, "rb")
    l = open(label_file, "rb")
    o = open(txt_file, "w")

    f.read(16)  # discard header info
    l.read(8)  # discard header info
    images = []

    for i in range(n_images):
        image = [ord(l.read(1))]  # get label
        for j in range(28 * 28):
            image.append(ord(f.read(1)))
        images.append(image)

    for image in images:
        o.write(",".join(str(pix) for pix in image) + "\n")

    f.close()
    o.close()
    l.close()


def convert2pc(in_file, out_folder, keyword):
    """ Convet MNIST data to point cloud """

    with open(in_file, 'r') as f:
        data = f.readlines()

    labels = []
    images = []
    for i in data:
        j = i.split(',')
        labels.append(j[0])
        images.append(j[1:])

    test = len(images[0])

    # process images according to 10 classes
    for i in tqdm(range(10)):
        indices = [ix for ix, x in enumerate(labels) if x == str(i)]

        for j in range(len(indices)):
            pc = []
            for r in range(28):
                for c in range(28):
                    if int(images[indices[j]][r * 28 + c]) > 0:
                        pc.append([c, r, 1.0])

            # write one image into one file
            if len(pc) >= 256:
                pc = random.sample(pc, 256)
            else:
                last_p = pc[-1]
                for p in range(256 - len(pc)):
                    pc.append(last_p)

            with open(out_folder + '/{}/{}_{}{}.txt'.format(i, i, j, keyword), 'w') as o:
                for pt in pc:
                    # x, y, z and normalized x, y, z
                    string = "{:.6f},{:.6f},{:.6f}".format(pt[0], pt[1], pt[2])
                    o.write(string + "\n")
            o.close()

    f.close()


def write_file_name(folder, out_file, keyword):
    """ Write file names under a folder into a txt """

    with open(out_file, 'w') as f:
        for i in range(10):  # write 10 classes separately
            for filename in glob.iglob(folder + '/{}/*'.format(i) + keyword + '.txt'):
                filename = filename.split('/')
                filename = filename[-1].split('\\')
                f.write(filename[-1][:-4] + "\n")
    f.close()


def main():
    """
    convert2txt("D:/Documents/Github/Pointnet_Pointnet2_pytorch/data/Fashion MNIST/train-images-idx3-ubyte",
                "D:/Documents/Github/Pointnet_Pointnet2_pytorch/data/Fashion MNIST/train-labels-idx1-ubyte",
                "D:/Documents/Github/Pointnet_Pointnet2_pytorch/data/Fashion MNIST/fashion_mnist_train.txt", 60000)
    convert2txt("D:/Documents/Github/Pointnet_Pointnet2_pytorch/data/Fashion MNIST/t10k-images-idx3-ubyte",
                "D:/Documents/Github/Pointnet_Pointnet2_pytorch/data/Fashion MNIST/t10k-labels-idx1-ubyte",
                "D:/Documents/Github/Pointnet_Pointnet2_pytorch/data/Fashion MNIST/fashion_mnist_test.txt", 10000)
    """

    for i in range(10):
        os.makedirs("D:/Documents/Github/Pointnet_Pointnet2_pytorch/data/Fashion MNIST/point_cloud/{}".format(i))

    convert2pc("D:/Documents/Github/Pointnet_Pointnet2_pytorch/data/Fashion MNIST/fashion_mnist_train.txt",
               "D:/Documents/Github/Pointnet_Pointnet2_pytorch/data/Fashion MNIST/point_cloud",
               'train')

    convert2pc("D:/Documents/Github/Pointnet_Pointnet2_pytorch/data/Fashion MNIST/fashion_mnist_test.txt",
               "D:/Documents/Github/Pointnet_Pointnet2_pytorch/data/Fashion MNIST/point_cloud",
               'test')

    write_file_name("D:/Documents/Github/Pointnet_Pointnet2_pytorch/data/Fashion MNIST/point_cloud",
                    "D:/Documents/Github/Pointnet_Pointnet2_pytorch/data/Fashion MNIST/point_cloud/fashion_mnist_train.txt",
                    'train')

    write_file_name("D:/Documents/Github/Pointnet_Pointnet2_pytorch/data/Fashion MNIST/point_cloud",
                    "D:/Documents/Github/Pointnet_Pointnet2_pytorch/data/Fashion MNIST/point_cloud/fashion_mnist_test.txt",
                    'test')

    with open("D:/Documents/Github/Pointnet_Pointnet2_pytorch/data/Fashion MNIST/point_cloud/filelist.txt", 'w') as f:
        for i in range(10):  # write 10 classes separately
            for filename in glob.iglob("D:/Documents/Github/Pointnet_Pointnet2_pytorch/data/Fashion MNIST/point_cloud"
                                       + '/{}/*.txt'.format(i)):
                filename = filename.split('/')
                filename = filename[-1].split('\\')
                f.write('{}/'.format(i) + filename[-1] + "\n")
    f.close()


if __name__ == '__main__':
    main()
