import numpy as np
import cv2
from sklearn import svm
from sklearn.externals import joblib
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation

from extra_tools import extract_HOG, chess_train

clf = chess_train('descriptors/HOG', ['pawn', 'king'], train_mod="SVC_linear")
print "Training done!"

# img = cv2.imread("/home/gerardo/Documents/ChessBaxter/chess_pieces/white_king1.png")
# img = cv2.imread("/home/gerardo/Documents/ChessBaxter/chess_pieces/window_132.jpeg")
img = cv2.imread("pawn.png")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# img = cv2.drawKeypoints(img,kp,img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# cv2.imwrite('sift_keypoints.jpg',img)
print "Compute descriptors"
kp, des1 = extract_HOG(gray, 0)
print "DESCRIPTORS:"
print len(des1)
# print des1
np.savetxt('test2.txt', des1, delimiter=',')

des1 = np.float32(des1)


# cv2.imshow('img', img)
# cv2.waitKey(0)

prediction = clf.predict(des1)
print prediction
pred_prob = clf.predict_proba(des1)
out = pred_prob.mean(axis=0)
print out

# total = 0
# counter = 0
# for entry in clf.predict_proba(des1):
# 	if prediction[counter] == 1:
# 		total = total + entry[0]
# 	else:
# 		total = total + entry[1]
# 	counter += 1
#
# print "Mean: " + str(total/counter)

# joblib.dump(clf,'classifiers/SIFT/dense/densepawnVking_classifier.pkl')
