import cv2
import numpy as np

img = cv2.imread('../../images/opencv-logo2.png', 0)
ret, thresh = cv2.threshold(img, 127, 155, 0)
contours, hierarchy = cv2.findContours(thresh, 1, 2)

cnt = contours[0]
M = cv2.moments(cnt)
print(M)

cx = int(M['m10'] / M['m00'])
cy = int(M['m01'] / M['m00'])
print('cx={0},cy={1}'.format(cx, cy))

# Contour Area = M['m00']
area = cv2.contourArea(cnt)
print('area=%f' % area)
print("M['m00']=%f" % M['m00'])

# Contour Perimeter 轮廓周长
Perimeter = cv2.arcLength(cnt, True)

# Contour Approximation
poly = cv2.imread('../../images/poly.png')
poly_gray = cv2.cvtColor(poly, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(poly_gray, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnt = contours[0]
epsilon = 0.01 * cv2.arcLength(cnt, True)
# epsilon = 0.1 * cv2.arcLength(cnt, True)
approx = cv2.approxPolyDP(cnt, epsilon, True)
cv2.drawContours(poly, [approx], 0, (0, 255, 0), 3)
# cv2.imshow('epsilon', poly)


# Convex Hull 凸检测
hand = cv2.imread('../../images/hand.jpg')
hand = cv2.resize(hand, (200, 300))
hand_gray = cv2.cvtColor(hand, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(hand_gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnt = contours[0]
hull = cv2.convexHull(cnt)
# Checking Convexity 检查凹凸性
print(cv2.isContourConvex(hull))
cv2.drawContours(hand, [hull], 0, (0, 255, 0), 3)
# cv2.imshow('hand', hand)


# Bounding Rectangle 对物体画边界矩形
# 分两种，第一种是外围
lightning = cv2.imread('../../images/lightning.jpg')
lightning_gray = cv2.cvtColor(lightning, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(lightning_gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnt = contours[0]
x, y, w, h = cv2.boundingRect(cnt)
print(x, y, w, h)
cv2.rectangle(lightning, (x, y), (x + w, y + h), (0, 255, 0), 2)

# 第二种是紧贴外围（旋转）
rect = cv2.minAreaRect(cnt)
box = cv2.boxPoints(rect)
box = np.int0(box)
cv2.drawContours(lightning, [box], 0, (0, 0, 255), 2)

# 画圆
(x, y), radius = cv2.minEnclosingCircle(cnt)
center = (int(x), int(y))
radius = int(radius)
cv2.circle(lightning, center, radius, (255, 0, 0), 2)

# Fitting an Ellipse
ellipse = cv2.fitEllipse(cnt)
cv2.ellipse(lightning, ellipse, (0, 255, 255), 2)

# Fitting a line
row, cols = img.shape[:2]
[vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0, 0.01, 0.01)
lefty = int((-x * vy / vx) + y)
righty = int(((cols - x) * vy / vx) + y)
cv2.line(lightning, (cols - 1, righty), (0, lefty), (255, 255, 0), 2)

cv2.imshow('lightning', lightning)
cv2.waitKey(0)
