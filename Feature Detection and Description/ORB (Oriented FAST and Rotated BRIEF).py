import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('../images/buildings.jpg', 0)
orb = cv2.ORB()
kp = orb.detect(img, None)
kp, des = orb.compute(img, kp)

img2 = cv2.drawKeypoints(img, kp, color=(0, 255, 0), flags=0)
plt.imshow(img2), plt.show()
