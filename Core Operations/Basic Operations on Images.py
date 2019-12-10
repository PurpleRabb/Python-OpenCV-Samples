import cv2
import numpy as np

img = cv2.imread('../images/buildings.jpg')
px = img[100, 100]  # B G R
print(px)

# pick blue
blue = img[100, 100, 0]
print(blue)

# modify pixel
img[100, 100] = [255, 255, 255]
print(img[100, 100])

# better methods to access pixel
print(img.item(100, 100, 2))  # Red
img.itemset((100, 100, 2), 100)
print(img.item(100, 100, 2))

print(img.shape, img.size, img.dtype)

_rect = img[200:300, 200:300]  # 选择200:300行、200:400列区域作为截取对象
img[0:100, 0:100] = _rect
# cv2.imshow('image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Splitting and Merging Image Channels
b, g, r = cv2.split(img)  # or b=img[:,:,0]
img = cv2.merge((b, g, r))

img[:, :, 2] = 0  # Make red 0
BLUE = [255, 0, 0]
constant = cv2.copyMakeBorder(img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=BLUE)
cv2.imshow('image', constant)
cv2.waitKey(0)
cv2.destroyAllWindows()
