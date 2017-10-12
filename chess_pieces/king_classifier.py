import numpy as np
import cv2
from sklearn import svm
from sklearn.externals import joblib
import pickle
import random


# Initialize an SVC classifier
clf = svm.SVC(probability=True,verbose=True)

folders_names = ['bishop','pawn','knight','queen','rook','square']

X = joblib.load('descriptors/SIFT/king/king.pkl')
num_of_positives = len(X)
print "Positives: " + str(num_of_positives)

negative_samples = []
for folder_name in folders_names:
    if len(negative_samples) == 0:
        negative_samples = joblib.load('descriptors/SIFT/'+folder_name+'/'+folder_name+'.pkl')
    else:
        negative_samples = np.concatenate((negative_samples,joblib.load('descriptors/SIFT/'+folder_name+'/'+folder_name+'.pkl')))

    print len(negative_samples)

negative_samples = random.sample(negative_samples, num_of_positives)
print len(negative_samples)
X = np.concatenate((X,negative_samples))
print len(X)

num_of_negatives = len(X) - num_of_positives

y = np.ones((num_of_positives, 1))
y = np.ravel(np.concatenate((y,np.zeros((num_of_negatives, 1)))))

print clf.fit(X,y)

joblib.dump(clf,'classifiers/new_SIFT/king_classifier.pkl')
joblib.dump(negative_samples,'descriptors/SIFT/king/negative_samples_king.pkl')
