import cv2
import numpy as np
from matplotlib import pyplot as plt

# 2D Convolution, 用指定kernel做卷积操作
img = cv2.imread('../images/opencv-logo2.png')
kernel = np.ones((5, 5), np.float32) / 25
print(kernel)
dst2d = cv2.filter2D(img, -1, kernel)
blur = cv2.blur(img, (5, 5))
gauss_blur = cv2.GaussianBlur(img, (5, 5), 0)
median = cv2.medianBlur(img, 5)
bil_blur = cv2.bilateralFilter(img, 9, 75, 75)  # 双边滤波

images = [img, dst2d, blur, gauss_blur, median, bil_blur]
titles = ['Original', 'Averaging', 'blur', 'gauss_blur', 'median_blur', 'bil_blur']

for i in range(5):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i])
    plt.title(titles[i])
plt.show()
