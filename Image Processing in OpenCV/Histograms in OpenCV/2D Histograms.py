import cv2
import numpy as np
from matplotlib import pyplot as plt

'''
1. 先将BGR转成HSV
2. [0,1] 处理H S通道
3. [180,256], 180 for H, 256 for S
4. 色调H从0-180，饱和度从0-256
'''

img = cv2.imread('../../images/buildings.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
# h, s, v = cv2.split(hsv)
# hist, xbins, ybins = np.histogram2d(h.ravel(), s.ravel(), [180, 256], [[0, 180], [0, 256]])

# 画图
# 1. cv2.imshow()
# 2. Matplotlib
plt.imshow(hist, interpolation='nearest')
plt.show()
