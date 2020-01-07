import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

imglist = []
binary_img_list = []

kernel = np.ones((1, 1), np.uint8)
kernel2 = cv2.getGaussianKernel(1, 0)


def find_images(path):
    if not os.path.exists(path):
        print('path error!')
    for root, dirs, names in os.walk(path):
        for filename in names:
            print(os.path.join(root, filename))
            img = cv2.imread(os.path.join(root, filename))
            bimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imglist.append(img)
            ret, bimg = cv2.threshold(bimg, 50, 255, cv2.THRESH_BINARY)
            bimg = cv2.medianBlur(bimg, 11)
            # bimg = cv2.GaussianBlur(bimg, (1, 1), 0)
            closing = cv2.erode(bimg, kernel)
            binary_img_list.append(closing)


sub = 1
find_images('../images/vcodes')
for (img, origin) in zip(binary_img_list, imglist):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    w_max, h_max, x_max, y_max = 0, 0, 0, 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w_max < w:
            w_max = w
            h_max = h
            x_max = x
            y_max = y
    cv2.rectangle(origin, (x_max, y_max), (x_max + w_max, y_max + h_max), (0, 255, 0), 2)
    plt.subplot(5, 2, sub)
    plt.axis('off')
    plt.imshow(origin)
    sub += 1
plt.show()
