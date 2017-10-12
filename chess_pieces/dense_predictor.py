import numpy as np
import cv2
from sklearn import svm
from sklearn.externals import joblib
from test_dense import myPredict


img = cv2.imread('window_132.jpeg')
print myPredict(img)

cv2.imshow('img',img)
cv2.waitKey(0)
