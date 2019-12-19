import cv2
import numpy as np

# 霍夫直线变换
src = cv2.imread('../images/buildings.jpg')
img = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
canny = cv2.Canny(img, 100, 150, apertureSize=3)

lines = cv2.HoughLines(canny, 1, np.pi / 360, 180)
for line in lines:
    for rho, theta in line:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv2.line(src, (x1, y1), (x2, y2), (0, 0, 255), 1)

cv2.imshow('img', src)
#cv2.waitKey(0)

# 概率霍夫变换
lines = cv2.HoughLinesP(canny, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
for line in lines:
    for x1, y1, x2, y2 in line:
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.imshow('img', img)
cv2.waitKey(0)
