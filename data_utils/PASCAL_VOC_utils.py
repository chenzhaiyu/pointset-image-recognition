""" Convert PASCAL VOC 2012 image dataset (semantic segmentation) to point cloud """

from tqdm import tqdm
import random
import numpy as np
from PIL import Image


def read_train_names(train_names_path):
    """
    Read the filenames of training images and labels into self.train_list
    """
    train_names = []
    f = open(train_names_path, 'r')
    line = None
    while 1:
        line = f.readline().replace('\n', '')
        if line is None or len(line) == 0:
            break
        train_names.append(line)
    f.close()
    return train_names


def read_val_names(val_names_path):
    """
    Read the filenames of validation images and labels into self.val_list
    """
    val_names = []
    f = open(val_names_path, 'r')
    line = None
    while 1:
        line = f.readline().replace('\n', '')
        if line is None or len(line) == 0:
            break
        val_names.append(line)
    f.close()
    return val_names


def get_points_label(image_path, mask_path, image_name):
    """
    Get point features and labels for points (pixels) in a single image
    A single image is considered as a single room
    """
    image = np.array(Image.open(image_path + image_name + '.jpg'))
    mask = np.array(Image.open(mask_path + image_name + '.png'))
    mask[mask > 20] = 0
    w, h = image.shape[0], image.shape[1]

    points = []
    labels = []
    for i in range(w):
        for j in range(h):
            if mask[i][j] != 0:
                points.append([i / 1000, j / 1000, 1, image[i][j][0], image[i][j][1], image[i][j][2]])
                labels.append(mask[i][j] - 1)  # TODO: change to S3DIS labels 0 - 19

    return np.array(points), np.array(labels)


def write_train_file(image_path, mask_path, train_names, out_path, num_sample):
    """
    Write training point clouds in to a numpy file
    All training data as an area
    """
    for train_name in tqdm(train_names):
        points, labels = get_points_label(image_path, mask_path, train_name)

        out_file = np.concatenate((points, labels.reshape((-1, 1))), axis=1)
        if out_file.shape[0] >= num_sample:
            indices = random.sample(range(out_file.shape[0]), num_sample)
            out_file = out_file[indices].astype(np.float64)
        else:
            last_p = out_file[-1, :]
            for p in range(num_sample - out_file.shape[0]):
                out_file = np.append(out_file, [last_p], axis=0)
        np.save(out_path + '/Area_1_' + train_name + '.npy', out_file)


def write_val_file(image_path, mask_path, val_names, out_path, num_sample):
    """
    Write validation point clouds in to a numpy file
    All validation data as an area
    """
    for val_name in tqdm(val_names):
        points, labels = get_points_label(image_path, mask_path, val_name)

        out_file = np.concatenate((points, labels.reshape((-1, 1))), axis=1)
        if out_file.shape[0] >= num_sample:
            indices = random.sample(range(out_file.shape[0]), num_sample)
            out_file = out_file[indices].astype(np.float64)
        else:
            last_p = out_file[-1, :]
            for p in range(num_sample - out_file.shape[0]):
                out_file = np.append(out_file, [last_p], axis=0)
        np.save(out_path + '/Area_2_' + val_name + '.npy', out_file)


def main():
    train_names = read_train_names(
        "D:/Documents/Datasets/VOCdevkit/VOC2012/ImageSets/Segmentation"
        "/train.txt")
    val_names = read_train_names(
        "D:/Documents/Datasets/VOCdevkit/VOC2012/ImageSets/Segmentation"
        "/val.txt")

    write_train_file("D:/Documents/Datasets/VOCdevkit/VOC2012/JPEGImages/",
                     "D:/Documents/Datasets/VOCdevkit/VOC2012/SegmentationClass/",
                     train_names,
                     "D:/Documents/Datasets/pascal_point_cloud",
                     4096)  # Area 1

    write_val_file("D:/Documents/Datasets/VOCdevkit/VOC2012/JPEGImages/",
                   "D:/Documents/Datasets/VOCdevkit/VOC2012/SegmentationClass/",
                   val_names,
                   "D:/Documents/Datasets/pascal_point_cloud",
                   4096)  # Area 2


if __name__ == '__main__':
    main()
