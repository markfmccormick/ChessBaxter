import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('kinect_images_new/new_chessboard.png',0)

keypoints = [[]]

with open('chessboard_keypoints.txt') as chessboard_keypoints:
    for line in chessboard_keypoints:
        line = line.strip('\n')
        line = line.split(',')
        keypoints[0].append([line[0], line[1]])

for point in keypoints[0]:
    cv2.circle(img, (int(point[0]),int(point[1])), 5, (255,0,0), -1)

cv2.imshow("test", img)
cv2.waitKey(0)
