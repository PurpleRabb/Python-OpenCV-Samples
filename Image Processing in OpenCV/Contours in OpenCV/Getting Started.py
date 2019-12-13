import cv2
import numpy as np

# 轮廓线：由图片中一些连续的点形成的图，这些点一般具有相同的颜色或者亮度，等高线图主要用来分析形状和物体侦测
# 为了准确性，一般用二值化的图像，可以借助阈值或者canny做边缘检测
# 等高线就像从黑色背景找白色，所以提前要把目标变成白色

image = cv2.imread('../../images/buildings.jpg')
im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, the = cv2.threshold(im_gray, 127, 255, 0)
contours, hierarchy = cv2.findContours(the, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# cv2.CHAIN_APPROX_SIMPLE 用来控制轮廓点的数量，如果是NONE则返回所有轮廓点，比如针对一条直线会返回所有直线上的点，
# 但是我们一般不需要这么多，SIMPLE则只返回两个点，可以节省内存

# 画出所有的等高线
cv2.drawContours(image, contours, -1, (0, 255, 0), 3)

# 画一个
cnt = contours[4]
# cv2.drawContours(image, [cnt], 0, (0, 255, 0), 3)
cv2.imshow('contours', image)
cv2.waitKey(0)
