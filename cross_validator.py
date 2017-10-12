import numpy as np
from sklearn.cross_validation import cross_val_score        # In some other versions of scikit-learn the package is called sklearn.model_selection
from sklearn import svm
from sklearn.externals import joblib

clf = svm.SVC(kernel='linear', C=1)

X = joblib.load('chess_pieces/descriptors/white_knight/white_knight.pkl')
num_of_positives = len(X)

folders_names = ['white_pawn','white_king','white_bishop','white_queen','white_rock', 'white_knight']

lengths = []
total_length = 0

for folder_name in folders_names:
    X = np.concatenate((X,joblib.load('chess_pieces/descriptors/'+folder_name+'/'+folder_name+'.pkl')))
    lengths.append(len(X)-total_length)
    total_length = len(X)


# b = [x*2 for x in a]          # TO MULTIPLY EVERY ELEMENT OF A LIST BY A CERTAIN FACTOR
c = 1
for length in lengths:
    if c == 1:
        y = np.ones((lengths[0],1))
        c += 1
    else:
        z = np.ones((length, 1))
        z = [x*c for x in z]
        print c
        print np.shape(y)
        print np.shape(z)
        y = np.concatenate((y,z))
        c += 1

y = np.ravel(y)
print y

# y = np.ones((num_of_positives, 1))
# y = np.ravel(np.concatenate((y,np.zeros((num_of_negatives, 1)))))

scores = cross_val_score(clf, X, y, cv=2)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
