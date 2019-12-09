import numpy as np
import cv2
from matplotlib import pyplot as plt

# 显示图像
# cv2.IMREAD_COLOR;cv2.IMREAD_GRAYSCALE;cv2.IMREAD_UNCHANGED
img = cv2.imread('../buildings.jpg', cv2.IMREAD_COLOR)
cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存图像
img = cv2.imread('../buildings.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imwrite('./buildings-gray.jpg', img)

# 用matplotlib显示图像
plt.imshow(img, cmap='gray', interpolation='bicubic')
plt.xticks([])
plt.yticks([])
plt.show()
