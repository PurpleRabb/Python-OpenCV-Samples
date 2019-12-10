import cv2
import numpy as np

flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
print(flags)

# 找出图片中的绿色目标

rgb_img = cv2.imread('../images/opencv-logo2.png')
hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2HSV)

# 可以通过以下方法找出RGB对应的HSV值
lower_green = np.uint8([[[0, 100, 0]]])
upper_green = np.uint8([[[0, 255, 0]]])
hsv_lower_green_test = cv2.cvtColor(lower_green, cv2.COLOR_BGR2HSV)
hsv_upper_green_test = cv2.cvtColor(upper_green, cv2.COLOR_BGR2HSV)
print(hsv_lower_green_test, hsv_upper_green_test)

# 设置HSV的阈值，取得MASK
hsv_lower_green = np.array([60, 255, 100])
hsv_upper_green = np.array([60, 255, 255])
mask = cv2.inRange(hsv_img, hsv_lower_green, hsv_upper_green)
result = cv2.bitwise_and(rgb_img, rgb_img, mask=mask)

cv2.imshow('result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
