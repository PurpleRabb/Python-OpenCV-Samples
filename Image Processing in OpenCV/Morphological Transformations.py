import cv2
import numpy as np
from matplotlib import pyplot as plt

# 图像形态学，典型的两种算法就是Erosion and Dilation(腐蚀与扩张),一般基于二值化的图像操作,腐蚀白的，扩张黑的

# 腐蚀，可以用来断开物体的连接
image = cv2.imread('../images/opencv-logo2.png', cv2.IMREAD_GRAYSCALE)
ret, bin_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
kernel = np.ones((5, 5), np.uint8)
bin_erode = cv2.erode(bin_image, kernel, iterations=1)

plt.subplot(221)
plt.imshow(image, 'gray')
plt.title('original')

plt.subplot(222)
plt.imshow(bin_image, 'gray')
plt.title('bin')

plt.subplot(223)
plt.imshow(bin_erode, 'gray')
plt.title('erode')

# Dilation 扩张
bin_dilation = cv2.dilate(bin_image, kernel)
plt.subplot(224)
plt.imshow(bin_dilation, 'gray')
plt.title('dilation')

# 图像的开闭运算,可以用来去噪

# 开运算 = 先腐蚀后扩张,去噪点有用
i_image = cv2.imread('../images/i-noisy.png', 0)
opening = cv2.morphologyEx(i_image, cv2.MORPH_OPEN, kernel)
cv2.imshow('open', opening)

# 闭运算 = 先扩张后腐蚀，可以用来闭合前景物体的一些洞或者去掉物体上的黑点
closing = cv2.morphologyEx(i_image, cv2.MORPH_CLOSE, kernel)

# 形态学梯度运算，相当于扩张与腐蚀之间的差值
gradient = cv2.morphologyEx(opening, cv2.MORPH_GRADIENT, kernel)
cv2.imshow('gradient', gradient)

# Top Hat, 原图像与开运算的区别
tophat = cv2.morphologyEx(i_image, cv2.MORPH_TOPHAT, kernel)
cv2.imshow('tophat', tophat)
# Black Hat， 原图像与闭运算的区别
blackhat = cv2.morphologyEx(i_image, cv2.MORPH_BLACKHAT, kernel)
cv2.imshow('blackhat', blackhat)
plt.show()

# 其他kernel
# Rectangular kernel
print(cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
# Elliptical kernel
print(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
# Cross-shaped kernel
print(cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5)))