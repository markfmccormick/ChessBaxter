import numpy as np
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread('kinect_images_new/new_chessboard.png',0)

keypoints = [[]]

with open('chessboard_keypoints.txt') as chessboard_keypoints:
    for line in chessboard_keypoints:
        line = line.strip('\n')
        line = line.split(',')
        keypoints[0].append([line[0], line[1]])

for point in keypoints[0]:
    cv2.circle(img1, (point[0],point[1]), 2, (255,0,0))
