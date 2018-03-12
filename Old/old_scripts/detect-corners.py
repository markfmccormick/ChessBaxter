import cv2
import numpy as np

filename = 'kinect_images_new/white_front/middle.jpeg'
img = cv2.imread(filename)
img = cv2.resize(img, (0,0), fx = 0.5, fy = 0.5)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,3,0.04)

#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.01*dst.max()]=[0,0,255]

cv2.imshow('dst',img)
cv2.waitKey()
