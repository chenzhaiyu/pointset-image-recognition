import numpy as np
from skimage import morphology
import cv2
from matplotlib import pyplot as plt


def grabcut(img, x0, y0, w, h):
    plt.imshow(img)
    plt.show()
    mask = np.zeros(img.shape[:2], np.uint8)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    rect = (x0, y0, w, h)

    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = img * mask2[:, :, np.newaxis]

    plt.imshow(img)
    # plt.colorbar()
    plt.show()

    return img


if __name__ == '__main__':
    img = cv2.imread("C:/Users/Bai/Downloads/grabcut/grabcut/opencv-python-foreground-extraction-tutorial.jpg")
    mask = grabcut(img, 161, 79, 150, 150)