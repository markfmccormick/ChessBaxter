import numpy as np
import cv2
from sklearn import svm
from sklearn.externals import joblib
import pickle
from sklearn import cross_validation

clf = svm.SVC(kernel='linear', C=1)

# Read all the positives
X = joblib.load('descriptors/SIFT/dense/pawn/pawn.pkl')
num_of_positives = len(X)

X = np.concatenate((X,joblib.load('descriptors/SIFT/dense/king/king.pkl')))

num_of_negatives = len(X) - num_of_positives

print len(X)
y = np.ones((num_of_positives, 1))
y = np.ravel(np.concatenate((y,np.zeros((num_of_negatives, 1)))))

print "..starting.."
scores = cross_validation.cross_val_score(clf, X, y, cv=3, verbose=10, n_jobs=3)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
