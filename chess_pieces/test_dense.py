import numpy as np
import cv2
from sklearn import svm
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


def myPredict(img1):

	dictionary = {0:"pawn", 1:"king"}
	siftdesc = cv2.xfeatures2d.SIFT_create()

	kp = dense_keypoints(img1)
	kp, des1 = siftdesc.compute(img1, kp)

	# If enough descriptors were found, run classifiers and choose the most confident
	if np.size(des1) > 50:

		### Load classifiers from a pickle file ###
		classifiers = []
		classifiers.append(joblib.load('classifiers/SIFT/dense/densepawnVking_classifier.pkl'))
		classifiers.append(joblib.load('classifiers/SIFT/dense/densekingVpawn_classifier.pkl'))


		max = float(0.0)
		piece_counter = 0
		piece = ""

		for classifier in classifiers:
			counter = 0
			total = 0
			prediction = classifier.predict(des1)
			# print prediction
			# print classifier.predict_proba(des1)
			for entry in classifier.predict_proba(des1):
				if piece_counter == 0:
					if prediction[counter] == 1:
						total = total + entry[0]
					else:
						total = total + entry[1]
					counter += 1
				elif piece_counter == 1:
					if prediction[counter] == 0:
						total = total + entry[0]
					else:
						total = total + entry[1]
					counter += 1

			confidence = total/counter
			# print "Mean: " + str(confidence)
			if confidence > max:
				max = confidence
				piece = piece_counter

			print dictionary[piece_counter] + " " + str(confidence)
			piece_counter += 1

		# If a piece was detected, return the name of that piece and the classifier's confidence

		return dictionary[piece], max

	return "None", 0
