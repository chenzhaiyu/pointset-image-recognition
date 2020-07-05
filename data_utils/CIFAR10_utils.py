""" Convert CIFAR10 image dataset to point cloud """

import os
from tqdm import tqdm
import glob
import random
import numpy as np
import pickle
from data_utils import grabcut


def write_train_data(cifar_folder, out_folder, num_sample):
    labels = []
    images = []
    filenames = []
    for i in range(1, 1 + 5):
        with open(cifar_folder + "data_batch_{}".format(i), "rb") as f:
            batch_i = pickle.load(f, encoding='bytes')
            labels.append(batch_i[b'labels'])
            images.append(batch_i[b'data'])
            filenames.append(batch_i[b'filenames'])
        f.close()

    labels = np.array(labels, dtype=np.int32)  #.reshape((-1, ))
    images = np.array(images, dtype=np.uint8)  #.reshape((-1, 32 * 32 * 3))
    filenames = np.array(filenames, dtype=np.str)  #.reshape((-1,))
    for i in tqdm(range(5)):
        # process images according to 10 classes
        for object_class in range(10):
            indices = [ix for ix, x in enumerate(labels[i, :]) if x == object_class]

            for j in range(len(indices)):
                image = np.zeros((32, 32, 3), dtype=np.uint8)  #images[i, indices[j], :].reshape((32, 32, 3))
                image[:, :, 0] = images[i, indices[j], 0:32 * 32].reshape((32, 32))
                image[:, :, 1] = images[i, indices[j], 32 * 32:32 * 32 * 2].reshape((32, 32))
                image[:, :, 2] = images[i, indices[j], 32 * 32 * 2:32 * 32 * 3].reshape((32, 32))
                foreground = grabcut.grabcut(image, x0=0, y0=0, w=31, h=31)

                if np.count_nonzero(foreground) / (32 * 32) > 0.1:  # ensure there are enough foreground points
                    pc = []
                    for r in range(32):
                        for c in range(32):
                            if foreground[r, c, 0] != 0 and foreground[r, c, 1] != 0 and foreground[r, c, 2] != 0:
                                pc.append([c, r, 1.0, image[r, c, 0], image[r, c, 1], image[r, c, 2]])  # x y z r g b

                    # randomly choose 256 points
                    pc_len = len(pc)
                    if pc_len >= num_sample:
                        pc = random.sample(pc, num_sample)
                    else:
                        last_p = pc[-1]
                        for p in range(num_sample - pc_len):
                            pc.append(last_p)

                    # write one image into one file
                    filename = filenames[i, indices[j]].split('_')
                    file_ext = filename[-1].split('.')
                    filename = np.concatenate((filename[0:-1], file_ext))
                    name = filename[0]
                    for k in filename[1:]:
                        name += k
                    with open(out_folder + '/{}/{}_{}train.txt'.format(object_class, object_class, name), 'w') as o:
                        for pt in pc:
                            # x, y, z and r, g, b
                            string = "{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f}".format(pt[0], pt[1], pt[2], pt[3], pt[4], pt[5])
                            o.write(string + "\n")
                    o.close()


def write_test_data(cifar_folder, out_folder, num_sample):
    labels = []
    images = []
    filenames = []
    with open(cifar_folder + "test_batch", "rb") as f:
        test_batch = pickle.load(f, encoding='bytes')
        labels.append(test_batch[b'labels'])
        images.append(test_batch[b'data'])
        filenames.append(test_batch[b'filenames'])
    f.close()

    labels = np.array(labels, dtype=np.int32)
    images = np.array(images, dtype=np.uint8)
    filenames = np.array(filenames, dtype=np.str)

    # process images according to 10 classes
    for object_class in tqdm(range(10)):
        indices = [ix for ix, x in enumerate(labels[0, :]) if x == object_class]

        for j in range(len(indices)):
            image = np.zeros((32, 32, 3), dtype=np.uint8)
            image[:, :, 0] = images[0, indices[j], 0:32 * 32].reshape((32, 32))
            image[:, :, 1] = images[0, indices[j], 32 * 32:32 * 32 * 2].reshape((32, 32))
            image[:, :, 2] = images[0, indices[j], 32 * 32 * 2:32 * 32 * 3].reshape((32, 32))
            foreground = grabcut.grabcut(image, x0=0, y0=0, w=31, h=31)

            if np.count_nonzero(foreground) / (32 * 32) > 0.1:  # ensure there are enough foreground points
                pc = []
                for r in range(32):
                    for c in range(32):
                        if foreground[r, c, 0] != 0 and foreground[r, c, 1] != 0 and foreground[r, c, 2] != 0:
                            pc.append([c, r, 1.0, image[r, c, 0], image[r, c, 1], image[r, c, 2]])  # x y z r g b

                # randomly choose 256 points
                pc_len = len(pc)
                if pc_len >= num_sample:
                    pc = random.sample(pc, num_sample)
                else:
                    last_p = pc[-1]
                    for p in range(num_sample - pc_len):
                        pc.append(last_p)

                # write one image into one file
                filename = filenames[0, indices[j]].split('_')
                file_ext = filename[-1].split('.')
                filename = np.concatenate((filename[0:-1], file_ext))
                name = filename[0]
                for k in filename[1:]:
                    name += k
                with open(out_folder + '/{}/{}_{}test.txt'.format(object_class, object_class, name), 'w') as o:
                    for pt in pc:
                        # x, y, z and r, g, b
                        string = "{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f}".format(pt[0], pt[1], pt[2], pt[3], pt[4], pt[5])
                        o.write(string + "\n")
                o.close()


def write_file_name(folder, out_file, keyword):
    """ Write file names under a folder into a txt """

    with open(out_file, 'w') as f:
        for i in range(10):  # write 10 classes separately
            for filename in glob.iglob(folder + '{}/*'.format(i) + keyword + '.txt'):
                filename = filename.split('/')
                filename = filename[-1].split('\\')
                f.write(filename[-1][:-4] + "\n")
    f.close()


if __name__ == '__main__':

    for i in range(10):
        os.makedirs("D:/Documents/Datasets/cifar-10-batches-py/point_cloud/{}".format(i))

    write_train_data("D:/Documents/Datasets/cifar-10-batches-py/",
                     "D:/Documents/Datasets/cifar-10-batches-py/point_cloud/",
                     256)
    write_test_data("D:/Documents/Datasets/cifar-10-batches-py/",
                    "D:/Documents/Datasets/cifar-10-batches-py/point_cloud/",
                    256)

    write_file_name("D:/Documents/Datasets/cifar-10-batches-py/point_cloud/",
                    "D:/Documents/Datasets/cifar-10-batches-py/point_cloud/cifar10_train.txt",
                    'train')

    write_file_name("D:/Documents/Datasets/cifar-10-batches-py/point_cloud/",
                    "D:/Documents/Datasets/cifar-10-batches-py/point_cloud/cifar10_test.txt",
                    'test')

    with open("D:/Documents/Datasets/cifar-10-batches-py/point_cloud/filelist.txt", 'w') as f:
        for i in range(10):  # write 10 classes separately
            for filename in glob.iglob("D:/Documents/Datasets/cifar-10-batches-py/point_cloud"
                                       + '/{}/*.txt'.format(i)):
                filename = filename.split('/')
                filename = filename[-1].split('\\')
                f.write('{}/'.format(i) + filename[-1] + "\n")
    f.close()