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
    plt.colorbar()
    plt.show()

    return img


def grabcut2(image, mask, x1=None, x2=None, y1=None, y2=None, mode="mask"):
    # Initialize _mask as sure background for GrabCut
    _mask = np.zeros(image.shape[:2], np.uint8)

    if mode == "skeleton":
        mask = morphology.skeletonize(mask.astype("uint8"))
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    if x1 is None:
        _mask[:, :] = cv2.GC_PR_BGD
    else:
        _mask[y1:y2, x1:x2] = cv2.GC_PR_BGD

    _mask[mask == True] = cv2.GC_FGD

    # GrabCut often gets assertion failure when image or mask size is weird
    try:
        _mask, bgd_model, fgd_model = cv2.grabCut(image, _mask.astype("uint8"), None, bgd_model, fgd_model, 2,
                                                  cv2.GC_INIT_WITH_MASK)

    except:
        print("Bug Occurs with cv2.grabCut!")
        pass

    # Covert _mask to human known format
    # Change sure background and probable background to 0
    _mask = 255 * np.where((_mask == 2) | (_mask == 0), 0, 1).astype('uint8')

    return _mask


if __name__ == '__main__':
    img = cv2.imread("C:/Users/Bai/Downloads/grabcut/grabcut/opencv-python-foreground-extraction-tutorial.jpg")
    mask = grabcut(img, 161, 79, 150, 150)