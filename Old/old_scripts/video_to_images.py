import cv2
import os
import fnmatch
from scipy import ndimage
import numpy as np

# Script for converting videos of chess pieces into images from the video frames
# to use as training data
# Constantly edited and adjusted to suit needs of different files

matches = []
for root, dirnames, filenames in os.walk('training_data/new_piece_data'):
    for filename in fnmatch.filter(filenames, '*.mp4'):
        matches.append(os.path.join(root, filename))

for filename in matches:

    if int(filename[29]) > -1:

        video = cv2.VideoCapture(filename)
        dest = filename[31:-4]
        dest = "training_data/piece_data/" + dest
        if not os.path.exists(dest):
            os.makedirs(dest)
        dest = dest+"/"+filename[29]
        print dest

        success,image = video.read()
        count = 0
        while success and count < 5000:
            if count %10 == 0:
                success,image = video.read()
                if np.shape(image) == (1080,1920,3):
                    image = image[300:-300, 600:-600]
                    image = np.rot90(image, 3)
                    cv2.imwrite(dest + "frame" + str(count) + ".jpg", image)
                elif np.shape(image) == (1920,1080,3):
                    image = image[600:-600, 300:-300]
                    cv2.imwrite(dest + "frame" + str(count) + ".jpg", image)
                # print "Saving frame " + str(count) + " to "+dest
                # image = ndimage.rotate(image, 270)
                # image = np.rot90(image, 3)
            count +=1
