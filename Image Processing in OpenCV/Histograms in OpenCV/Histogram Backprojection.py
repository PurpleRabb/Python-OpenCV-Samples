import cv2
import numpy as np
from matplotlib import pyplot as plt

# np.set_printoptions(threshold=np.inf)

'''
直方图反投影：如果一幅图像的区域中显示的是一种机构纹理或者一个独特物体，
那么这个区域的直方图可以看作一个概率函数，它给的是某个像素属于该纹理或物体的概率。
所谓反向投影就是首先计算某一特征的直方图模型，然后使用模型去寻找测试图像中存在的该特征
'''

'''Numpy算法实现>>>>>>>>>>>>>>>>>'''
# 1. 首先计算需要寻找的目标物体的颜色直方图M和整体图片的直方图I
roi = cv2.imread('../../images/flower-part.png')
hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

target = cv2.imread('../../images/flower-full.jpg')
hsvt = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
M = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
I = cv2.calcHist([hsvt], [0, 1], None, [180, 256], [0, 180, 0, 256])

# 2. 计算比率 R=M/I, 反投影R, 把R看作调色板，创建一幅图像，图像的每个像素作为其对应的目标概率
# B(x,y) = R[h(x,y),s(x,y)], h=hue,s=saturation
R = M / (I + 1)
h, s, v = cv2.split(hsvt)
B = R[h.ravel(), s.ravel()]
B = np.minimum(B, 1)
B = B.reshape(hsvt.shape[:2])

# 3. 创建并应用卷积核
disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # 5*5椭圆卷积核
cv2.filter2D(B, -1, disc, B)
B = np.uint8(B)
cv2.normalize(B, B, 0, 255, cv2.NORM_MINMAX)

# 4. 最大强度的位置给出了物体的位置
ret, thresh = cv2.threshold(B, 50, 255, 0)
cv2.imshow('res', thresh)
cv2.waitKey(0)
'''<<<<<<<<<<<<<<<<<<<<<<<<<Numpy算法实现'''

# Backprojection in OpenCV
# 从物体局部找到整个物体
roi = cv2.imread('../../images/flower-part.png')
hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

target = cv2.imread('../../images/flower-full.jpg')
hsvt = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)

roihist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
cv2.normalize(roihist, roihist, 0, 255, cv2.NORM_MINMAX)
dst = cv2.calcBackProject([hsvt], [0, 1], roihist, [0, 180, 0, 256], 1)

disc = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
cv2.filter2D(dst, -1, disc, dst)

ret, thresh = cv2.threshold(dst, 50, 255, 0)
thresh = cv2.merge((thresh, thresh, thresh))
res = cv2.bitwise_and(target, thresh)
res = np.vstack((target, thresh, res))
cv2.imshow('res', res)
cv2.waitKey(0)
