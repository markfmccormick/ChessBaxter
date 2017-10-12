import numpy as np
import cv2
from sklearn import svm
from sklearn.externals import joblib
import pickle


# Initialize an SVC classifier
clf = svm.SVC(probability=True,verbose=True)

X = joblib.load('descriptors/SIFT/king/king.pkl')
num_of_positives = len(X)
print num_of_positives

X = np.concatenate((X,joblib.load('descriptors/SIFT/pawn/pawn.pkl')))


num_of_negatives = len(X) - num_of_positives

y = np.ones((num_of_positives, 1))
y = np.ravel(np.concatenate((y,np.zeros((num_of_negatives, 1)))))

print clf.fit(X,y)

img = cv2.imread("white_king8.png")
siftdesc = cv2.xfeatures2d.SIFT_create()
kp1, des1 = siftdesc.detectAndCompute(img, None)
print clf.predict(des1)
print clf.predict_proba(des1)

joblib.dump(clf,'classifiers/SIFT/kingVpawn_classifier.pkl')
