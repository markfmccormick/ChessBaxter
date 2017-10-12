import numpy as np
import cv2, os
import pickle
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
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

def sliding_window(image, stepSize, windowSize):
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def extract_HOG(img, class_id):
	gray = img #cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	winSize = 11

	feat = []
	desc = []
	nlevel = 1
	while gray.shape[0] >= winSize*2.5 or gray.shape[1] >= winSize*2.5:

		for (x, y, window) in sliding_window(gray, stepSize=5, windowSize=(winSize, winSize)):
			if window.shape[0] != winSize or window.shape[1] != winSize:
				continue

			feat.append([x,y, nlevel, class_id])
			desc.append(hog(window).tolist())

		gray = cv2.pyrDown(gray)
		nlevel += 1

	return feat, desc

def hog(img):
        bin_n = 16
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)
        bins = np.int32(bin_n * ang / (2 * np.pi))  # quantizing binvalues in (0...16)
        bin_cells = bins[:10, :10], bins[10:, :10], bins[:10, 10:], bins[10:, 10:]
        mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
        hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
        hist = np.hstack(hists)  # hist is a 64 bit vector
        # hist = hist / np.linalg.norm(hist)
        hist = normalize(hist[:,np.newaxis], axis=0).ravel()

        return hist

def chess_train(out_dir_desc, train_classes, train_mod="LR_l1"):
	'''

	Args:
		out_dir_desc: path where image descriptors are saved
		train_classes: classes used to train the classifier
		train_mod: (string) Classification model to be used, options are:
				   LR_l1: L1 Logistic Regression
				   LR_l2: L2 Logistic Regression
				   LR_liblinear: liblinear Logistic Regression
				   SVC_linear: linear SVC
				   KNN: KNeighborsClassifier

	Returns:
		classification_model: trained classification model

	'''

	print("**************** Training classification model")
	all_desc = []
	all_class_no = []
	for i, class_name in enumerate(train_classes):
		filename = os.path.join(out_dir_desc, class_name, class_name + ".vocab")
		desc = np.float32(np.array(np.loadtxt(filename, delimiter=',')))
		class_no = np.ones((len(desc), 1)) * i
		all_desc = _loop_list(desc, all_desc)
		for d in class_no:
			all_class_no.append(d[0])

	if train_mod == "LR_l1":
		classification_model = LogisticRegression(C=1, penalty='l1')
	elif train_mod == "LR_l2":
		classification_model = LogisticRegression(C=1, penalty='l2')
	elif train_mod == "LR_liblinear":
		classification_model = LogisticRegression(C=1, solver='liblinear')
	elif train_mod == "SVC_linear":
		classification_model = SVC(kernel='linear', C=1, probability=True, random_state=0)
	elif train_mod == "KNN":
		classification_model = KNeighborsClassifier()
	else:
		raise("Unknown classification model")

	scores = cross_validation.cross_val_score(classification_model, all_desc, all_class_no, cv = 10)
	print("Cross-validation results")
	print(scores)
	print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

	classification_model.fit(all_desc, all_class_no)
	ans = classification_model.score(all_desc, all_class_no)
	print(ans)

	print("**************** Done!")

	return classification_model


def _loop_list(in_list, out_list):
	for d in in_list:
		out_list.append(d)

	return out_list
