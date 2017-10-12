import numpy as np
import cv2
from sklearn import svm
from sklearn.externals import joblib
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation

# Initialize an SVC classifier
clf = svm.SVC(probability=True,verbose=True)
# clf = LogisticRegression(C=1, penalty='l1')

X = joblib.load('descriptors/SIFT/pawn/pawn.pkl')
XXX = np.copy(X)
num_of_positives = len(X)
print num_of_positives

X = np.concatenate((X,joblib.load('descriptors/SIFT/king/king.pkl')))


num_of_negatives = len(X) - num_of_positives

y = np.ones((num_of_positives, 1))
y = np.ravel(np.concatenate((y,np.zeros((num_of_negatives, 1)))))

scores = cross_validation.cross_val_score(clf, X, y, cv = 10)
print("Cross-validation results")
print(scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

print clf.fit(X,y)
img = cv2.imread("white_pawn27.png")
siftdesc = cv2.xfeatures2d.SIFT_create()
kp1, des1 = siftdesc.detectAndCompute(img, None)
# for des in des1:
#     # print len(X[num_of_positives:])
#     # print len(X[:num_of_positives])
#     if des in X[:num_of_positives]:
#         print "YES"
#     else:
#         print "no"
print clf.predict(des1)
print clf.predict_proba(des1)
#
# total = 0
# for entry in clf.predict_proba(des1):
#     print entry
# print np.mean(clf.predict_proba(des1)[0][0])

# print clf.score(des1[range(0,3)],labels[range(0,3)])

joblib.dump(clf,'classifiers/SIFT/pawnVking_classifier.pkl')
