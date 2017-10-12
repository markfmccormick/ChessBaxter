import numpy as np
import cv2
from sklearn import svm
from sklearn.externals import joblib
import pickle


# Initialize an SVC classifier
clf = svm.SVC(probability=True,verbose=True)

folders_names = ['bishop','king','knight','pawn','queen','rook']

X = joblib.load('descriptors/SIFT/square/square.pkl')
num_of_positives = len(X)

for folder_name in folders_names:
    X = np.concatenate((X,joblib.load('descriptors/SIFT/'+folder_name+'/'+folder_name+'.pkl')))


num_of_negatives = len(X) - num_of_positives

y = np.ones((num_of_positives, 1))
y = np.ravel(np.concatenate((y,np.zeros((num_of_negatives, 1)))))

print clf.fit(X,y)

joblib.dump(clf,'classifiers/SIFT/square_classifier.pkl')
