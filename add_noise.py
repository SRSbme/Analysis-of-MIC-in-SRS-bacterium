import os
import cv2
import random
import numpy as np


def sp_noise(image, prob=0.5):
    output = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                print(0, i, j)
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
                print(255,i,j)
            else:
                print(True)
                output[i][j] = image[i][j]
    return output


def add_noise(input_dir='datasets/original_img', output_dir='datasets/noise_im'):
    os.makedirs(output_dir, exist_ok=True)
    imgs_path = os.listdir(input_dir)
    print(len(imgs_path))
    for img_path in imgs_path:
        print(img_path)
        path = os.path.join(input_dir, img_path)
        img = cv2.imread(path, -1)
        img1 = sp_noise(img)
        img1 = img+img1
        # cv2.imwrite(os.path.join(output_dir, img_path), img1)
        imgs = np.hstack((img, img1))
        cv2.imshow('img', imgs)
        cv2.waitKey(0)
        # cv2.destroyAllWindows()

if __name__ == '__main__':
    add_noise()
