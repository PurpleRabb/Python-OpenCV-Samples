import cv2
import numpy as np
from matplotlib import pyplot as plt

# flatten()和ravel()函数都能将多维数组降为一维，区别在于numpy.flatten()返回是拷贝，
# 对拷贝所做的修改不会影响原始矩阵，而numpy.ravel()返回的是视图,修改视图，原始矩阵也会受到影响

img = cv2.imread('../../images/buildings.jpg', 0)
hist, bins = np.histogram(img.flatten(), 256, [0, 256])
# print(hist)
# print(bins)
cdf = hist.cumsum()  # 按列累加
cdf_normalized = cdf * hist.max() / cdf.max()  # 标准化

cdf_m = np.ma.masked_equal(cdf, 0)  # cdf为原数组，当数组元素为0时，掩盖(计算时被忽略)
cdf_m = ((cdf_m - cdf_m.min()) / (cdf_m.max() - cdf_m.min())) * 255  # 归一化
cdf = np.ma.filled(cdf_m).astype('uint8')
img2 = cdf[img]
# print(img2.ravel())
cv2.imshow('img2', img2)
# print(cdf_m)

plt.plot(cdf_normalized, color='b')
plt.hist(img.flatten(), 256, [0, 256], color='r')
plt.xlim([0, 256])
plt.legend(('cdf', 'hist'), loc='upper left')
plt.show()

# OpenCV 的API均衡化直方图
equ = cv2.equalizeHist(img)
res = np.hstack((img, equ))
# cv2.imshow('res', res)


# CLAHE (Contrast Limited Adaptive Histogram Equalization)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
cl1 = clahe.apply(img)
cv2.imshow('clahe', cl1)
res = np.hstack((res, cl1))

cv2.imshow('res', res)
cv2.waitKey(0)
