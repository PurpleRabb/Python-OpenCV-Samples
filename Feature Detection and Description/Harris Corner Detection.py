import cv2
import numpy as np

# Find Harris corners
img = cv2.imread('../images/arrows.jpg')
img1 = img.copy()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv2.cornerHarris(gray, 2, 3, 0.04)
dst = cv2.dilate(dst, None)
print(dst.max())
img[dst > (0.01 * dst.max())] = [0, 0, 255]
cv2.imshow('dst', img)
cv2.waitKey(0)

# find Harris corners
dst1 = cv2.cornerHarris(gray, 2, 3, 0.04)
dst1 = cv2.dilate(dst1, None)
res, dst1 = cv2.threshold(dst1, 0.01 * dst1.max(), 255, 0)
dst1 = np.uint8(dst1)

# find centroids
res, labels, stats, centroids = cv2.connectedComponentsWithStats(dst1)

# define the criteria to stop and refine the corners
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.01)
corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)

# draw
res = np.hstack((centroids, corners))
res = np.int0(res)
img1[res[:, 1], res[:, 0]] = [0, 0, 255]
img1[res[:, 3], res[:, 2]] = [0, 255, 0]
cv2.imshow('dst1', img1)
cv2.waitKey(0)
