import cv2
from sklearn.externals import joblib
import numpy as np
import glob
from extra_tools import extract_HOG

# Initiate HOG detector
hog = cv2.HOGDescriptor()


folders_names = ['bishop','king','knight','pawn','queen','rook']

print "\nNumber of descriptors:\n"
# Go through each folder containing pictures of the chess pieces
for folder_name in folders_names:
	# Read all images in the current folder
	images = []
	for filename in glob.glob('cropped_pictures/'+folder_name+'/*.png'):
		images.append(cv2.imread(filename))

	descriptors = []


	for image in images:
		# cv2.imshow('',image)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()
		# Detect and compute key points and descriptors for the image
		kp1, des1 = extract_HOG(image, 0)
		for des in des1:
			descriptors.append(des)
	print folder_name + ": " + str(len(descriptors))

	joblib.dump(descriptors,'descriptors_test/HOG/'+folder_name+'/'+folder_name+'.pkl')
