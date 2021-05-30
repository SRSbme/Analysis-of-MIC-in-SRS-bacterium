import cv2
import numpy as np


def watershed(img):
    blurred = cv2.pyrMeanShiftFiltering(img, 10, 100)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    ret, binnary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kerhel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mb = cv2.morphologyEx(binnary, cv2.MORPH_OPEN, kerhel, iterations=2)
    sure_bg = cv2.dilate(mb, kerhel, iterations=3)

    dist = cv2.distanceTransform(mb, cv2.DIST_L2, 3)
    # dist_optput = cv2.normalize(dist, 0, 1.0, cv2.NORM_MINMAX)
    ret, surface = cv2.threshold(dist, dist.max() * 0.6, 255, cv2.THRESH_BINARY)

    surface_fg = np.uint8(surface)
    unknown = cv2.subtract(sure_bg, surface_fg)
    ret, markers = cv2.connectedComponents(surface_fg)

    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(img, markers=markers)
    img[markers == -1] = [0, 0, 255]
    return img


if __name__ == '__main__':
    img = cv2.imread('1.png', 1)
    print(img.shape)  # [h,w,3]
    img = watershed(img)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
