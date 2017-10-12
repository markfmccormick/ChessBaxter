import numpy as np
import cv2
import pickle
import glob
from sklearn.externals import joblib


# Initiate SIFT detector
siftdesc = cv2.xfeatures2d.SIFT_create()


folders_names = ['bishop','king','knight','pawn','queen','rook','square']

print "\nNumber of descriptors:\n"
# Go through each folder containing pictures of the chess pieces
for folder_name in folders_names:
    # Read all images in the current folder
    images = []
    for filename in glob.glob('cropped_pictures/'+folder_name+'/*.png'):
        images.append(cv2.imread(filename,0))

    descriptors = []

    for image in images:
        # Detect and compute key points and descriptors for the image
        kp1, des1 = siftdesc.detectAndCompute(image, None)

        for c, des in enumerate(des1):
            # if c < 50:
            descriptors.append(des)
    print folder_name + ": " + str(len(descriptors))

    joblib.dump(descriptors,'descriptors/SIFT/'+folder_name+'/'+folder_name+'.pkl')
