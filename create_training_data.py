import cv2
import os
import random
import fnmatch
from scipy import ndimage
import numpy as np

# Script for converting videos of chess pieces into images from the video frames
# to use as training data
# Adjusted regularly to update hard-coded values it heavily uses

# Function which blackens out the background of the image to leave only the piece
# Based on: https://docs.opencv.org/3.1.0/d8/d83/tutorial_py_grabcut.html
def isolate_piece(img):
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    # (x,y,w,h) Form a square around the piece, big enough for all the pieces
    rect = (40,40,380,620)

    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,10,cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img = img*mask2[:,:,np.newaxis]
    return img

os.mkdir("training_data/In_progress")

matches = []
for root, dirnames, filenames in os.walk('training_data/new_piece_data'):
    for filename in fnmatch.filter(filenames, '*.mp4'):
        matches.append(os.path.join(root, filename))

for filename in matches:
    # was 29, now 15
    if int(filename[29]) > -1:
        video = cv2.VideoCapture(filename)
        dest = filename[31:-4]
        dest = "training_data/piece_data/" + dest
        if not os.path.exists(dest):
            os.makedirs(dest)
        dest = dest+"/"
        print dest + filename[29]

        success,image = video.read()
        count = 0
        while success and count < 500:
            # Randomise name, model training splits data based on filename
            name = str(int(random.random()*1000000000))+".jpg"
            success,image = video.read()
            if count %5 == 0:
                if np.shape(image) == (1080,1920,3):
                    image = image[300:-300, 600:-600]
                    image = np.rot90(image, 3)
                    image = isolate_piece(image)
                    cv2.imwrite(dest + name, image)
                elif np.shape(image) == (1920,1080,3):
                    image = image[600:-600, 300:-300]
                    image = isolate_piece(image)
                    cv2.imwrite(dest + name, image)
            count +=1

os.rmdir("training_data/In_progress")
