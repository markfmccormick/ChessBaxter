import numpy as np
import cv2
from sklearn import svm
from sklearn.externals import joblib
from extra_tools_test import get_X_and_y, get_X_and_y_pawn_king


#########################
# PAWN VS REST

X,y = get_X_and_y('SIFT')

# Initialize an SVC classifier
clf1 = svm.SVC(probability=True,verbose=True)

clf1.fit(X,y)

joblib.dump(clf1,'classifiers_test/SIFT/pawnVall_svm_classifier.pkl')

#########################
# PAWN VS KING

X,y = get_X_and_y_pawn_king('SIFT')

# Initialize an SVC classifier
clf2 = svm.SVC(probability=True,verbose=True)

clf2.fit(X,y)

joblib.dump(clf2,'classifiers_test/SIFT/pawnVking_svm_classifier.pkl')
