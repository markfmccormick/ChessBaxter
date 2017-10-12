from chess_pieces.extra_tools import hog
import numpy as np
import cv2
import glob
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

def grid_search(X, y, parameters, model):

	print("...Calculating the best parameters...")

	# Split the dataset into two parts
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

	# Create the correspondent classifier   - LR || KNN -
	if model == "LR":
		clf = GridSearchCV(LogisticRegression(), parameters, cv=5)
	elif model == "KNN":
		clf = GridSearchCV(KNeighborsClassifier(), parameters, cv=5)
	else:
		raise ValueError('Not a valid model name')

	clf.fit(X_train, y_train)

	print("Best parameters set found on development set:")
	print(clf.best_params_)

	print("Detailed classification report:")
	y_true, y_pred = y_test, clf.predict(X_test)
	print(classification_report(y_true, y_pred))

	return clf.best_params_


########
######## Create a classifier with colour histograms to recognise black and white pieces
######## class 0 = black, class 1 = white

folders_names = ['whites', 'blacks']
whites = []
blacks = []


for folder_name in folders_names:

# for filename in glob.glob('chess_pieces/cropped_pictures/' + folder_name + '/*.png'):
	for filename in glob.glob('chess_pieces/colour_sliding_windows/'+folder_name+'/*.jpg'):

		image = cv2.imread(filename)
		# gray = image.copy() #cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		
		if folder_name == "whites":
			whites.append(image)
		else:
			blacks.append(image)

print "hi",len(whites)
print len(blacks)
X = []
y = []

white_counter = 0
while white_counter < 24:

	hist = hog(whites[white_counter])
	X.append(hist)
	y.append(1)
	white_counter += 1


for black in blacks:

	hist = hog(black)
	X.append(hist)
	y.append(0)

# print clf.fit(X,y)
lr_parameters = [{'penalty': ['l2'], 'C': [1, 10, 100, 1000], 'solver': ['lbfgs']},
				 {'penalty': ['l2'], 'C': [1, 10, 100, 1000], 'solver': ['liblinear']},
				 {'penalty': ['l2'], 'C': [1, 10, 100, 1000], 'solver': ['newton-cg']},
				 {'penalty': ['l1'], 'C': [1, 10, 100, 1000], 'solver': ['liblinear']}]

# k = 5,9,13,17, ..., 101
k = range(5,127,4)
knn_parameters = {'n_neighbors': k}
# best_parameters = grid_search(X, y, knn_parameters, model='KNN')
best_parameters = grid_search(X, y, lr_parameters, model='LR')


colour_clf = LogisticRegression(penalty=best_parameters['penalty'],C=best_parameters['C'])
# colour_clf = KNeighborsClassifier(n_neighbors=100)

scores = cross_val_score(colour_clf,X,y,verbose=1)
print("Cross-validation results")
print(scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

colour_clf.fit(X,y)

joblib.dump(colour_clf,'chess_pieces/classifiers/colour_hist/black_white_classifier.pkl')
