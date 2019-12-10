import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('../images/opencv-logo2.png')
print(img.shape)
res = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

# OR
height, width = img.shape[:2]
res = cv2.resize(img, (2 * width, 2 * height), interpolation=cv2.INTER_CUBIC)

print(res.shape)

plt.subplot(331)
plt.imshow(img)
plt.subplot(332)
plt.imshow(res)

# Translation
# 平移矩阵
plt.subplot(333)
rows, cols = img.shape[:2]
M = np.float32([[1, 0, 100], [0, 1, 50]])
dst = cv2.warpAffine(img, M, (cols, rows))
plt.imshow(dst)

# Rotation
# 旋转矩阵
plt.subplot(334)
M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 30, 1)
dst = cv2.warpAffine(img, M, (cols, rows))
plt.imshow(dst)

# Affine Transformation 仿射变换: 变换前的平行线变换后依然平行，所以找三个点就可以了
plt.subplot(335)
pts1 = np.float32([[50, 50], [200, 50], [50, 200]])  # 选三个原点
pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
M = cv2.getAffineTransform(pts1, pts2)
dst = cv2.warpAffine(img, M, (cols, rows))
plt.imshow(dst)

# Perspective Transformation 透视变换：变换前的是直线的，变换后还是直线，不会歪斜，所以需要四个标准点
plt.subplot(336)
pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
M = cv2.getPerspectiveTransform(pts1, pts2)
dst = cv2.warpPerspective(img, M, (300, 300))
plt.imshow(dst)

plt.show()
