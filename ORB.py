import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('/home/mufengjun260/PycharmProjects/SLAM-Study/Sample1.jpg', 0)

orb = cv2.ORB_create()

kp = orb.detect(img, None)  # 描述符

kp, des = orb.compute(img, kp)

img = cv2.drawKeypoints(img, kp, img, color=(255, 0, 0))  # 画到img上面

plt.imshow(img)
plt.show()