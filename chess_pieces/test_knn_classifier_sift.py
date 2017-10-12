import numpy as np
import cv2
from sklearn import svm
from sklearn.externals import joblib
from extra_tools_test import get_X_and_y, get_X_and_y_pawn_king, my_grid_search
from sklearn.neighbors import KNeighborsClassifier

#########################
# PAWN VS REST

X,y = get_X_and_y('SIFT')

k = range(5,102,10)
parameters = {'n_neighbors': k}

best_parameters = my_grid_search(X, y, parameters, model='KNN')

clf1 = KNeighborsClassifier(n_neighbors=best_parameters['n_neighbors'])

clf1.fit(X,y)

joblib.dump(clf1,'classifiers_test/SIFT/pawnVall_knn_classifier.pkl')

#########################
# PAWN VS KING

X,y = get_X_and_y_pawn_king('SIFT')

k = range(5,102,10)
parameters = {'n_neighbors': k}

best_parameters = my_grid_search(X, y, parameters, model='KNN')

clf2 = KNeighborsClassifier(n_neighbors=best_parameters['n_neighbors'])

clf2.fit(X,y)

joblib.dump(clf2,'classifiers_test/SIFT/pawnVking_knn_classifier.pkl')
