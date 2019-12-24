import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('../images/cells.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(img.shape)
Gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
Gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
Gxy = cv2.sqrt(Gx * Gx + Gy * Gy)
Gxy2 = Gxy.copy()
print(Gxy.shape, Gx.shape, Gy.shape)

y, x = Gxy.shape

# 非极大值抑制
# [g1 g2 g3]
# [g4 g0 g5]
# [g6 g7 g8]

for i in range(1, y - 1):
    for j in range(1, x - 1):

        if Gxy[i, j] == 0:
            Gxy2[i, j] = 0
        else:
            gradX = Gx[i, j]
            gradY = Gy[i, j]
            gradTemp = Gxy[i, j]

            # 如果Y方向幅度值较大
            if np.abs(gradY) > np.abs(gradX):
                weight = abs(gradX / gradY)
                grad2 = Gxy[i - 1, j]
                grad4 = Gxy[i + 1, j]
                # 如果x,y方向梯度符号相同
                if gradX * gradY > 0:
                    grad1 = Gxy[i - 1, j - 1]
                    grad3 = Gxy[i + 1, j + 1]
                # 如果x,y方向梯度符号相反
                else:
                    grad1 = Gxy[i - 1, j + 1]
                    grad3 = Gxy[i + 1, j - 1]

            # 如果X方向幅度值较大
            else:
                weight = abs(gradY / gradX)
                grad2 = Gxy[i, j - 1]
                grad4 = Gxy[i, j + 1]
                # 如果x,y方向梯度符号相同
                if gradX * gradY > 0:
                    grad1 = Gxy[i + 1, j - 1]
                    grad3 = Gxy[i - 1, j + 1]
                # 如果x,y方向梯度符号相反
                else:
                    grad1 = Gxy[i - 1, j - 1]
                    grad3 = Gxy[i + 1, j + 1]

            gradTemp1 = weight * grad1 + (1 - weight) * grad2
            gradTemp2 = weight * grad3 + (1 - weight) * grad4
            if gradTemp >= gradTemp1 and gradTemp >= gradTemp2:
                Gxy2[i, j] = gradTemp
            else:
                Gxy2[i, j] = 0

# 双阈值检测与边缘连接
DT = np.zeros([y, x])
TL = 0.3 * np.max(Gxy2)
TH = 0.5 * np.max(Gxy2)

for i in range(1, y - 1):
    for j in range(1, x - 1):
        if Gxy2[i, j] < TL:
            DT[i, j] = 0
        elif Gxy2[i, j] > TH:
            DT[i, j] = 1
        elif ((Gxy2[i - 1, j - 1:j + 1] < TH).any() or (Gxy2[i + 1, j - 1:j + 1]).any()
              or (Gxy2[i, [j - 1, j + 1]] < TH).any()):
            DT[i, j] = 1

plt.subplot(221)
plt.axis('off')
plt.imshow(Gx, 'gray')
plt.subplot(222)
plt.axis('off')
plt.imshow(Gy, 'gray')
plt.subplot(223)
plt.axis('off')
plt.imshow(Gxy, 'gray')

plt.subplot(224)
plt.axis('off')
plt.imshow(DT, 'gray')

plt.show()
