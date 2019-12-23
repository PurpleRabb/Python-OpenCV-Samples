import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('../images/box.png', 0)

fast = cv2.FastFeatureDetector()

kp = fast.detect(img, None)
img2 = cv2.drawKeypoints(img, kp, color=(255, 0, 0))

print('Threshold: ', fast.getInt('threshold'))
print('nonmaxSuppression: ', fast.getBool('nonmaxSuppression'))
print('neighborhood:', fast.getInt('type'))
print('Total keypoints with nonmaxSuppression:', len(kp))

cv2.imshow('fast', img2)
cv2.waitKey(0)
