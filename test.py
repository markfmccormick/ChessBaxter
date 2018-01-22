import numpy as np
import cv2

def create_chesssquares(keypoints):
    keypoints = keypoints.reshape(9,9,2)
    positions = []
    for y in range(8):
        for x in range(8):
            square = [[],[]]
            square[0].append(keypoints[y][x])
            square[0].append(keypoints[y][x+1])
            square[1].append(keypoints[y+1][x])
            square[1].append(keypoints[y+1][x+1])
            positions.append(square)
    return np.array(positions, dtype="float32")

keypoints = [[]]
with open('chessboard_keypoints.txt') as chessboard_keypoints:
    for line in chessboard_keypoints:
        line = line.strip('\n')
        line = line.split(',')
        keypoints[0].append([line[0], line[1]])

keypoints = np.array(keypoints, dtype="float32")

board = create_chesssquares(keypoints)

imgpath = "kinect_images_new/white_front/middle.jpeg"
img = cv2.imread(imgpath)
print np.shape(img)
