""" Generate translated CIFAR10 image and point cloud samples """

import pickle
import numpy as np
from PIL import Image
import random
from matplotlib import pyplot as plt
import glob


def get_image(cifar_folder, out_folder, num_sample):
    """
    Get images from original CIFAR10 dataset
    :param cifar_folder: path of CIFAR10 dataset
    :param out_folder: output folder path for generated images
    :param num_sample: number of generated images
    """
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

    for n in range(num_sample):
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        j = np.random.choice(len(labels[0, :]))  # randomly choose an image in this batch

        image[:, :, 0] = images[0, j, 0:32 * 32].reshape((32, 32))
        image[:, :, 1] = images[0, j, 32 * 32:32 * 32 * 2].reshape((32, 32))
        image[:, :, 2] = images[0, j, 32 * 32 * 2:32 * 32 * 3].reshape((32, 32))

        # save the image
        im = Image.fromarray(image)
        im.save(out_folder + "{}_".format(labels[0, j]) + filenames[0, j])


def diagonal_shift(filename, out_folder, delta):
    """
    Achieve diagonal shift for an image -> 25 x 25 image size
    :param filename: image path
    :param out_folder: output folder path for generated images
    :param delta: shift amount [pixels]
    """
    assert (delta <= 11)

    im = np.array(Image.open(filename))
    # padding the original image to 36 x 36
    im_36 = np.pad(im, ((2, 2), (2, 2), (0, 0)), 'edge')

    image = im_36[delta:25 + delta, delta:25 + delta, :]

    # save the image
    im = Image.fromarray(image)
    object_name = filename.split('/')
    im.save(out_folder + "shift{}_".format(delta) + object_name[-1])


def generate_point_cloud(filename, out_folder, num_points):
    """
    Generate point cloud for an image
    :param filename: image path
    :param out_folder: output folder path for generated point cloud
    :param num_points: number of points in the point cloud
    """
    image = np.array(Image.open(filename))

    pc = []
    for r in range(image.shape[0]):
        for c in range(image.shape[1]):
            pc.append([c, r, 1.0, image[r, c, 0], image[r, c, 1], image[r, c, 2]])  # x y z r g b

    # randomly choose 256 points
    pc_len = len(pc)
    if pc_len >= num_points:
        pc = random.sample(pc, num_points)
    else:
        last_p = pc[-1]
        for p in range(num_points - pc_len):
            pc.append(last_p)

    # write one image into one file
    object_name = filename.split('/')
    object_name = object_name[-1].split('.')[0]

    with open(out_folder + '{}.txt'.format(object_name), 'w') as o:
        for pt in pc:
            # x, y, z and r, g, b
            string = "{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f}".format(pt[0], pt[1], pt[2], pt[3], pt[4],
                                                                        pt[5])
            o.write(string + "\n")
    o.close()


if __name__ == '__main__':
    # get_image("D:/Documents/Datasets/cifar-10-batches-py/",
    #           "D:/Documents/Datasets/cifar-10-batches-py/translation_samples/",
    #           10)

    # for filename in glob.iglob('D:/Documents/Datasets/cifar-10-batches-py/translation_samples/*.png'):
    #     diagonal_shift(filename,
    #                    "D:/Documents/Datasets/cifar-10-batches-py/translation_samples/shifted_images_7/", 7)

    # diagonal_shift('D:/Documents/Datasets/cifar-10-batches-py/translation_samples/0_biplane_s_000764.png',
    #                'D:/Documents/Datasets/cifar-10-batches-py/translation_samples/shifted_biplane/', 8)

    # for filename in glob.iglob('D:/Documents/Datasets/cifar-10-batches-py/translation_samples/shifted_images_7/*.png'):
    #     generate_point_cloud(filename,
    #                          "D:/Documents/Datasets/cifar-10-batches-py/translation_samples/shifted_point_cloud/",
    #                          512)

    # generate_point_cloud('D:/Documents/Datasets/cifar-10-batches-py/translation_samples/9_fire_truck_s_001064.png',
    #                      "D:/Documents/Datasets/cifar-10-batches-py/translation_samples/visualization/",
    #                      512)

    # jet_prob = np.array([0.21585518, 0.3345137, 0.9867716, 0.541304, 0.6591645, 0.7680666, 0.8154752, 0.38577095]) * 100
    jet_prob = np.array([0.2165115, 0.30921245, 0.2212514, 0.5479918, 0.22948211, 0.7701827, 0.67735416, 0.8736717, 0.4541842]) * 100
    vgg = np.array([0.1, 0.02, 0.94, 0.99, 0.31, 0.08, 0.67, 0.9, 0.07]) * 100
    # vgg = np.array([0.1, 0.02, 0.94, 0.99, 0.31, 0.08, 0.67, 0.9]) * 100
    plt.figure()
    plt.plot(np.arange(0, 9), jet_prob)
    plt.plot(np.arange(0, 9), vgg)
    plt.xlabel('Number of shifted pixels', fontsize=16)
    plt.ylabel('Probability of being "airplane" (%)', fontsize=16)
    plt.ylim([0.0, 100.0])
    plt.legend(['PointCNN', 'VGG16'])
    plt.show()
