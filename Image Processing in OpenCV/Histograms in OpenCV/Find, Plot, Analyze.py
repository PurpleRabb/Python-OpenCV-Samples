import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('../../images/buildings.jpg', 0)

# Histogram Calculation in OpenCV
hist = cv2.calcHist([img], [0], None, [256], [0, 256])

# Histogram Calculation in Numpy
hist, bins = np.histogram(img.ravel(), 256, [0, 256])

# 绘制直方图，两种方法：

# 简单方法用Matplotlib
plt.hist(img.ravel(), 256, [0, 256])

# 彩色直方图
img = cv2.imread('../../images/buildings.jpg')
color = ('b', 'g', 'r')
for i, col in enumerate(color):
    histr = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(histr, color=col)
    plt.xlim([0, 256])
plt.show()

# OpenCV方式，可以用模板采集某一部分的直方图
plt.figure()
img = cv2.imread('../../images/buildings.jpg', 0)
# 创建模板
mask = np.zeros(img.shape[:2], np.uint8)
mask[100:300, 100:400] = 255
masked_img = cv2.bitwise_and(img, img, mask=mask)

hist_full = cv2.calcHist([img], [0], None, [256], [0, 256])
hist_mask = cv2.calcHist([img], [0], mask, [256], [0, 256])

plt.subplot(221), plt.imshow(img, 'gray')
plt.subplot(222), plt.imshow(mask, 'gray')
plt.subplot(223), plt.imshow(masked_img, 'gray')
plt.subplot(224), plt.plot(hist_full), plt.plot(hist_mask)
plt.xlim([0, 256])

plt.show()
