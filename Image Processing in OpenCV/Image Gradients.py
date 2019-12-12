import cv2
from matplotlib import pyplot as plt
import numpy as np

# 图像梯度运算(边缘检测): (1) 找出图像边界 (2) 三个算子Sobel,Scharr,Laplacian

img = cv2.imread('../images/box.png', 0)
laplacian = cv2.Laplacian(img, cv2.CV_64F)  # 拉普拉斯
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
sobelxy = cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=3)

images = [img, laplacian, sobelx, sobely, sobelxy]
titles = ['origin', 'laplacian', 'sobelx', 'sobely', 'sobelxy']
print(sobelxy.dtype)
for i in range(5):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i])
plt.show()

# 注意转换数据类型，黑到白的sobel是正值，白到黑的sobel是负值，负值转换成uin8的时候会变成0，导致边界丢失
img = cv2.imread('../images/box.png', 0)

# Output dtype = cv2.CV_8U
# sobelx8u = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=5)
sobelx8u = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=5)

# Output dtype = cv2.CV_64F. Then take its absolute and convert to cv2.CV_8U
sobelx64f = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=5)

abs_sobel64f = np.absolute(sobelx64f)
sobel_8u = np.uint8(abs_sobel64f)

plt.subplot(1, 3, 1), plt.imshow(img, cmap='gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(1, 3, 2), plt.imshow(sobelx8u, cmap='gray')
plt.title('Sobel CV_8U'), plt.xticks([]), plt.yticks([])
plt.subplot(1, 3, 3), plt.imshow(sobel_8u, cmap='gray')
plt.title('Sobel abs(CV_64F)'), plt.xticks([]), plt.yticks([])

plt.show()

# 另一种操作方法：
x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
absX = cv2.convertScaleAbs(x)  # 转回uint8
absY = cv2.convertScaleAbs(y)
dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
cv2.imshow("absX", absX)
cv2.imshow("absY", absY)
cv2.imshow("Result", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
