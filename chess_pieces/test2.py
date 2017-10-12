import numpy as np
import cv2
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score, ShuffleSplit
from extra_tools_test import get_X_and_y, get_X_and_y_pawn_king
from sklearn.externals import joblib


X,y = get_X_and_y('ORB')


clf = joblib.load('classifiers_test/ORB/pawnVall_svm_classifier.pkl')

scores = cross_val_score(clf, X, y)
print("Cross-validation results")
print(scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))




X,y = get_X_and_y_pawn_king('ORB')

clf = joblib.load('classifiers_test/ORB/pawnVking_svm_classifier.pkl')

scores = cross_val_score(clf, X, y)
print("Cross-validation results")
print(scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
