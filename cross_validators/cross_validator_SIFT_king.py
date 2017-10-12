import numpy as np
from sklearn.cross_validation import cross_val_score        # In some other versions of scikit-learn the package is called sklearn.model_selection
from sklearn import svm
from sklearn.externals import joblib

clf = svm.SVC(kernel='linear', C=1)

# Read all the positives
X = joblib.load('../chess_pieces/descriptors/SIFT/king/king.pkl')
num_of_positives = len(X)
X = np.concatenate((X,joblib.load('../chess_pieces/descriptors/SIFT/king/negative_samples_king.pkl')))

num_of_negatives = len(X) - num_of_positives

y = np.ones((num_of_positives, 1))

y = np.ravel(np.concatenate((y,np.zeros((num_of_negatives, 1)))))

scores = cross_val_score(clf, X, y, cv=5, verbose=10, n_jobs=2)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
