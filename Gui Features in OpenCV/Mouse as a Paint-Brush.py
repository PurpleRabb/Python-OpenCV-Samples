import cv2
import numpy as np

# 查看事件类型
events = [i for i in dir(cv2) if 'EVENT' in i]
print(events)
img = np.zeros((512, 512, 3), np.uint8)
drawing = False
ix, iy = -1, -1


def draw_circle(event, x, y, flags, param):
    global drawing, ix, iy
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
    elif event == cv2.EVENT_RBUTTONUP:
        img.fill(0)


cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_circle)
while True:
    cv2.imshow('image', img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:  # ESC
        break

cv2.destroyAllWindows()
