import numpy as np
import cv2
from sklearn import svm
from sklearn.externals import joblib
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation

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


# Initialize an SVC classifier
clf = svm.SVC(probability=True,verbose=True)

siftdesc = cv2.xfeatures2d.SIFT_create()

X = joblib.load('descriptors/SIFT/dense/king/king.pkl')
num_of_positives = len(X)
print num_of_positives

X = np.concatenate((X,joblib.load('descriptors/SIFT/dense/pawn/pawn.pkl')))
num_of_negatives = len(X) - num_of_positives

y = np.zeros((num_of_positives, 1))
y = np.ravel(np.concatenate((y,np.ones((num_of_negatives, 1)))))

print clf.fit(X,y)

joblib.dump(clf,'classifiers/SIFT/dense/densekingVpawn_classifier.pkl')

### SECOND PART ###

siftdesc = cv2.xfeatures2d.SIFT_create()

img = cv2.imread("white_king8.png")

kp = dense_keypoints(img)
kp, des1 = siftdesc.compute(img, kp)


cv2.imshow('img', img)
cv2.waitKey(0)

prediction = clf.predict(des1)
print prediction
print clf.predict_proba(des1)[0][0]

total = 0
counter = 0
for entry in clf.predict_proba(des1):
	if prediction[counter] == 0:
		total = total + entry[1]
	else:
		total = total + entry[0]
	counter += 1

print "Mean: " + str(total/counter)
