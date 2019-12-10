import cv2
import numpy as np

e1 = cv2.getTickCount()
img = cv2.imread('opencv-logo2.png')
for i in range(5, 49, 2):
    img = cv2.medianBlur(img, i)
e2 = cv2.getTickCount()
time = (e2 - e1) / cv2.getTickFrequency()
print("time:%f" % time)
print(cv2.useOptimized())

#ipython环境下可以用 %timeit 测试执行速度
