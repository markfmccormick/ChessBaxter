import numpy as np
import cv2
import glob
import os
import fnmatch

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

# for filename in glob.glob('training_data/new_piece_data/**/*.mp4', recursive=True):
#     print filename[31:-4]

# matches = []
# for root, dirnames, filenames in os.walk('training_data/new_piece_data'):
#     for filename in fnmatch.filter(filenames, '*.mp4'):
#         matches.append(os.path.join(root, filename))
#
# print matches[0][31:-4]

video = cv2.VideoCapture("training_data/new_piece_data/4/black_bishop.mp4")

success, image = video.read()
count = 0
print video.get(cv2.CAP_PROP_FPS)
print video.get(cv2.CAP_PROP_FRAME_COUNT)
while count < 2500 and success:
    success, image = video.read()
    count += 1

print count
cv2.imshow("image", image)
print np.shape(image)
cv2.waitKey(0)