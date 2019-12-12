import cv2
from matplotlib import pyplot as plt

# Canny边缘检测算法
image = cv2.imread('../images/lion.jpg', 0)
canny = cv2.Canny(image, 120, 240)
plt.subplot(121), plt.imshow(image, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(canny, cmap='gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()
