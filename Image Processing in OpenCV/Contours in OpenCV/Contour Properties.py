import cv2
import numpy as np

lightning = cv2.imread('../../images/lightning.jpg')
lightning_gray = cv2.cvtColor(lightning, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(lightning_gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnt = contours[0]
cv2.drawContours(lightning, [cnt], 0, (0, 255, 0), 2)

# 纵横比 aspect ratio
x, y, w, h = cv2.boundingRect(cnt)
cv2.rectangle(lightning, (x, y), (x + w, y + h), (0, 255, 0), 2)
aspect_ratio = float(w) / h
print(aspect_ratio)

# Extent = (Object Area)/(Bounding Rectangle Area)
area = cv2.contourArea(cnt)
rect_area = w * h
extent = float(area) / rect_area
print(extent)

# solidity
hull = cv2.convexHull(cnt)
hull_area = cv2.contourArea(hull)
cv2.drawContours(lightning, [hull], 0, [255, 0, 0], 2)
solidity = float(area) / hull_area
print(solidity)

# Equivalent Diameter
equi_diameter = np.sqrt(4 * area / np.pi)

# Orientation
(x, y), (MA, ma), angle = cv2.fitEllipse(cnt)

# Mask and Pixel Point
mask = np.zeros(lightning_gray.shape, np.uint8)
cv2.drawContours(mask, [cnt], 0, 255, -1)
# 两个方法返回的坐标表示不一样
pixelpoints = np.transpose(np.nonzero(mask))
pixelpoints2 = cv2.findNonZero(mask)
cv2.drawContours(lightning, [pixelpoints2], 0, (255, 255, 0), 3)
# print(pixelpoints)
# print(pixelpoints2)

# Maximum Value, Minimum Value and their locations
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(lightning_gray, mask=mask)
print(min_val, max_val, min_loc, max_loc)

# Mean Color or Mean Intensity
mean_val = cv2.mean(lightning_gray, mask=mask)
print(mean_val)

# Extreme Points 极点
leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])
print(leftmost, rightmost, topmost, bottommost)
cv2.circle(lightning, leftmost, 3, (0, 0, 255), 5)
cv2.circle(lightning, rightmost, 3, (0, 0, 255), 5)
cv2.circle(lightning, topmost, 3, (0, 0, 255), 5)
cv2.circle(lightning, bottommost, 3, (0, 0, 255), 5)
cv2.imshow('contours', lightning)
cv2.waitKey(0)
