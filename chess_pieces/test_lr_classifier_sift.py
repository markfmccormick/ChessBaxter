import numpy as np
import cv2
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score, ShuffleSplit
from sklearn.metrics import classification_report
from extra_tools_test import get_X_and_y, get_X_and_y_pawn_king, my_grid_search


parameters = [{'penalty': ['l2'], 'C': [0.1, 1, 10, 100, 1000]},
			  {'penalty': ['l1'], 'C': [0.1, 1, 10, 100, 1000]}]

#########################
# PAWN VS THE REST

X,y = get_X_and_y('SIFT')

best_parameters = my_grid_search(X, y, parameters,model='LR')
# Initialize an SVC classifier
clf1 = LogisticRegression(C=best_parameters['C'], penalty=best_parameters['penalty'])

print clf1.fit(X,y)

joblib.dump(clf1,'classifiers_test/SIFT/pawnVall_lr_classifier.pkl')

#########################
# PAWN VS KING

X,y = get_X_and_y_pawn_king('SIFT')
best_parameters = my_grid_search(X, y, parameters,model='LR')
# Initialize an SVC classifier
clf2 = LogisticRegression(C=best_parameters['C'], penalty=best_parameters['penalty'])

print clf2.fit(X,y)

joblib.dump(clf2,'classifiers_test/SIFT/pawnVking_lr_classifier.pkl')
