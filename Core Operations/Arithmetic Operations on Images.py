import numpy as np
import cv2

# Image Addition
x = np.uint8([250])
y = np.uint8([10])

print(cv2.add(x, y))
print(x + y)

# Image Blending
img1 = cv2.imread('../images/opencv-logo2.png')
img2 = cv2.imread('../images/buildings.jpg')

img1 = cv2.resize(img1, (200, 200))
img2 = cv2.resize(img2, (200, 200))
dst = cv2.addWeighted(img1, 0.1, img2, 0.9, 0)
# cv2.imshow('image', dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# Bitwise Operations
img1 = cv2.imread('../images/buildings.jpg')
img2 = cv2.imread('../images/opencv-logo2.png')
img2 = cv2.resize(img2, (100, 100))
rows, cols, channels = img2.shape
roi = img1[0:rows, 0:cols]

# 创建mask
img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
# cv2.imshow('img2gray', img2gray)  # logo转灰度
ret, mask = cv2.threshold(img2gray, 150, 255, cv2.THRESH_BINARY)  # 二值化
mask_inv = cv2.bitwise_not(mask)  # 翻转
# cv2.imshow('mask_inv', mask_inv)
# cv2.imshow('mask', mask)

img1_bg = cv2.bitwise_and(roi, roi, mask=mask)
# cv2.imshow('img1_bg', img1_bg)
img2_fg = cv2.bitwise_and(img2, img2, mask=mask_inv)
# cv2.imshow('img2_fg', img2_fg)

dst = cv2.add(img1_bg, img2_fg)
img1[0:rows, 0:cols] = dst
cv2.imshow('res', img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
