import numpy as np
import cv2
import pickle
import glob
from sklearn.externals import joblib

def dense_keypoints(img, scaleLevels=1, scaleFactor=1.2, varyStepWithScale=False):
	curScale = 1.0
	curStep = 5
	curBound = 5
	featureScaleMul = 1/scaleFactor
	kp = []
	for curLevel in range(0, scaleLevels):
		for x in range(curBound, img.shape[1] - curBound, curStep):
			for y in range(curBound, img.shape[0] - curBound, curStep):
				kp.append(cv2.KeyPoint(x, y, curScale, _class_id=0))

		curScale *= featureScaleMul
		if varyStepWithScale:
			curStep = curStep * featureScaleMul + 0.5

	return kp

# Initiate SIFT detector
siftdesc = cv2.xfeatures2d.SIFT_create()


folders_names = ['bishop','king','knight','pawn','queen','rook','square']

print "\nNumber of descriptors:\n"
# Go through each folder containing pictures of the chess pieces
for folder_name in folders_names:
	# Read all images in the current folder
	images = []
	for filename in glob.glob('cropped_pictures/'+folder_name+'/*.png'):
		images.append(cv2.imread(filename))

	descriptors = []

	boo = False
	for image in images:
		# Detect and compute dense key points and descriptors for the image
		kp = dense_keypoints(image)
		print "Number of keypoints: " + str(len(kp))
		# image = cv2.drawKeypoints(image,kp,image,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
		kp1, des1 = siftdesc.compute(image, kp)
		# cv2.imshow('dense features', image)
		# cv2.waitKey(0)
		if len(des1) == 60:
			if boo == False:
				boo = True
			else:
				print des1
				np.savetxt('test.txt', des1, delimiter=',')
				kp2 = dense_keypoints(image)
				kp3, des2 = siftdesc.compute(image, kp2)
				np.savetxt('test0.txt', des2, delimiter=',')
		for c, des in enumerate(des1):
			descriptors.append(des)


	print folder_name + "'s total number of descriptors': " + str(len(descriptors))

	joblib.dump(descriptors,'descriptors/SIFT/dense/'+folder_name+'/'+folder_name+'.pkl')
