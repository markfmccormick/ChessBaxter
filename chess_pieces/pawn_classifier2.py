import numpy as np
import cv2
from sklearn import svm
from sklearn.externals import joblib
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from extra_tools import extract_HOG, chess_train

def pawn_prediction(gray):
	clf = chess_train('chess_pieces/descriptors/HOG', ['pawn', 'king','queen','knight','bishop','rook'], train_mod="SVC_linear")
	# print "Training done!"

	# cv2.imshow('',gray)
	# cv2.waitKey(0)
	# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	# print "Compute descriptors"
	kp, des1 = extract_HOG(gray, 0)

	np.savetxt('test2.txt', des1, delimiter=',')

	des1 = np.float32(des1)


	prediction = clf.predict(des1)
	# print prediction
	pred_prob = clf.predict_proba(des1)
	# print pred_prob

	values = []
	counter = 0

	for entry in pred_prob:
		if float(prediction[counter]) == 0.0:
			values.append(entry[0])
		counter += 1


	out = pred_prob.mean(axis=0)
	# print "Out: " + str(out)
	if out[0] > out[1]:
		return "pawn", out[0]
	else:
		return "the rest", out[1]
