import cv2
import numpy as np

'''
1. Convexity defects and how to find them.
2. Finding shortest distance from a point to a polygon
3. Matching different shapes
'''
# [1]
lightning = cv2.imread('../../images/lightning.jpg')
lightning_gray = cv2.cvtColor(lightning, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(lightning_gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnt = contours[0]
hull = cv2.convexHull(cnt, returnPoints=False)
defects = cv2.convexityDefects(cnt, hull)
print(cnt.shape)
print(cnt)
print(defects.shape)
# 返回N行，每行的含义  [ start point, end point, farthest point, approximate distance to farthest point ]
for i in range(defects.shape[0]):
    s, e, f, d = defects[i, 0]
    start = tuple(cnt[s][0])
    end = tuple(cnt[e][0])
    far = tuple(cnt[f][0])
    cv2.line(lightning, start, end, [0, 255, 0], 2)
    cv2.circle(lightning, far, 5, [0, 0, 255], -1)
cv2.imshow('img', lightning)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# [2]
dist = cv2.pointPolygonTest(cnt, (100, 50), True)
print(dist)

# [3] 对旋转不是很敏感
circle = cv2.imread('../../images/circle.png', cv2.IMREAD_GRAYSCALE)
hexagon = cv2.imread('../../images/hexagon.png', cv2.IMREAD_GRAYSCALE)
square = cv2.imread('../../images/square.png', cv2.IMREAD_GRAYSCALE)

ret, thresh = cv2.threshold(circle, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
ret, thresh2 = cv2.threshold(hexagon, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
ret, thresh3 = cv2.threshold(square, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

cv2.imshow('circle', thresh)
cv2.imshow('hexagon', thresh2)
cv2.imshow('square', thresh3)

contours, hierarchy = cv2.findContours(thresh, 2, 1)
cnt1 = contours[0]
contours, hierarchy = cv2.findContours(thresh2, 2, 1)
cnt2 = contours[0]
contours, hierarchy = cv2.findContours(thresh3, 2, 1)
cnt3 = contours[0]

ret1 = cv2.matchShapes(cnt1, cnt2, 1, 0.0)
ret2 = cv2.matchShapes(cnt2, cnt3, 1, 0.0)
ret3 = cv2.matchShapes(cnt1, cnt3, 1, 0.0)
print(ret1, ret2, ret3)
cv2.waitKey(0)
