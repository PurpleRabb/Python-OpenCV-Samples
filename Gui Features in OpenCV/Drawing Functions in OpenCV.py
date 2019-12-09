import cv2
import numpy as np

img = np.zeros((512, 512, 3), np.uint8)
# img.fill(255)
cv2.line(img, (0, 0), (200, 300), (255, 0, 0), 5)
cv2.rectangle(img, (200, 300), (500, 500), (0, 255, 0), 3)
cv2.circle(img, (256, 256), 30, (0, 0, 255), 2)
cv2.ellipse(img, (256, 256), (100, 50), 0, 0, 360, 255)

# 多边形绘制，指定int32的顶点
pts = np.array([[10, 5], [20, 30], [70, 20], [50, 10]], np.int32)
cv2.polylines(img, [pts], True, (0, 255, 255), 3)

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, "Draw Text", (10, 500), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
