import cv2
from matplotlib import pyplot as plt

idcard = cv2.imread('C:\\Users\\liushuo\\Desktop\\photo.jpg')
h, w = idcard.shape[:2]
print(w, h)
resize = ()
kernel_size = ()
if w < 1000:
    resize = (w, h)
    kernel_size = (8, 8)
else:
    resize = (int(w / 2), int(h / 2))
    kernel_size = (21, 21)

s_idcard = cv2.resize(idcard, resize)
s_idcard = cv2.GaussianBlur(s_idcard, (3, 3), 0)
s_idcard_gray = cv2.cvtColor(s_idcard, cv2.COLOR_BGR2GRAY)
res, s_idcard_bin = cv2.threshold(s_idcard_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

canny = cv2.Canny(s_idcard_bin, 10, 15)  # 20是最小阈值,50是最大阈值 边缘检测
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
dilation = cv2.dilate(canny, kernel, iterations=1)  # 膨胀一下，来连接边缘

contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

rois = []
for i in range(len(contours)):
    cnt = contours[i]
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(s_idcard, (x, y), (x + w, y + h), (0, 255, 0), 2)
    print(x, y, w, h)
    roi = s_idcard[y:y + h, x:x + w]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    rois.append(roi)

roi_num = len(rois)

for i in range(roi_num):
    plt.subplot(2, roi_num / 2 + 1, i + 1)
    plt.axis('off')
    plt.imshow(rois[i])

plt.show()
cv2.imshow('id', s_idcard)
cv2.waitKey(0)
