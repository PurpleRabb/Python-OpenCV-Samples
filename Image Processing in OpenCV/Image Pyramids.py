import cv2
import numpy as np

# 图像金字塔

# 高斯金字塔：用于下采样。高斯金字塔是最基本的图像塔。
# 原理：首先将原图像作为最底层图像G0（高斯金字塔的第0层），利用高斯核（5*5）对其进行卷积，然后对卷积后的图像进行下采样（去除偶数行和列）得到上一层图像G1，
# 将此图像作为输入，重复卷积和下采样操作得到更上一层图像，反复迭代多次，形成一个金字塔形的图像数据结构，即高斯金字塔。

img = cv2.imread("../images/buildings.jpg")
lower_reso = cv2.pyrDown(img)  # 下采样，图像变为原来的四分之一
higher_reso1 = cv2.pyrUp(lower_reso)  # 上采样，还原图像，但是因为做了高斯模糊，还原的图像是模糊的

# 拉普拉斯金字塔：用于重建图像，也就是预测残差，对图像进行最大程度的还原。比如一幅小图像重建为一幅大图，
# 原理：用高斯金字塔的每一层图像减去其上一层图像上采样，得到一系列的差值图像即为 LP 分解图像。
l_image = cv2.subtract(img, higher_reso1)

cv2.imshow('original', img)
cv2.imshow('reso1', lower_reso)
cv2.imshow('reso1-up', higher_reso1)
cv2.imshow('laps', l_image)
# cv2.waitKey(0)

# 一个图像混合的小例子: 生成新水果

apple = cv2.imread('../images/apple.png')
orange = cv2.imread('../images/orange.png')
# 对apple做n次高斯金字塔
G = apple.copy()
gpA = [G]
for i in range(5):
    G = cv2.pyrDown(G)
    gpA.append(G)

# 对orange做n次高斯金字塔
G = orange.copy()
gpB = [G]
for i in range(5):
    G = cv2.pyrDown(G)
    gpB.append(G)

# 生成apple的拉普拉斯金字塔
lpA = [gpA[4]]
for i in range(4, 0, -1):
    GE = cv2.pyrUp(gpA[i])
    L = cv2.subtract(gpA[i - 1], GE)
    lpA.append(L)

# 生成orange的拉普拉斯金字塔
lpB = [gpB[4]]
for i in range(4, 0, -1):
    GE = cv2.pyrUp(gpB[i])
    L = cv2.subtract(gpB[i - 1], GE)
    lpB.append(L)

LS = []
# 图片左右相加
for la, lb in zip(lpA, lpB):
    rows, cols, dpt = la.shape
    ls = np.hstack((la[:, 0:cols // 2], lb[:, cols // 2:]))
    LS.append(ls)

ls_ = LS[0]
for i in range(1, 5):
    ls_ = cv2.pyrUp(ls_)
    ls_ = cv2.add(ls_, LS[i])

cv2.imshow('blending', ls_)
cv2.waitKey(0)
cv2.destroyAllWindows()
