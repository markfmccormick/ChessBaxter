import numpy as np
import cv2, os, imutils
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report
from skimage import feature

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.externals import joblib

import csv
import random
import numpy as np
from sklearn import svm, cross_validation
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier

#
# Find the best parameters for an LR or KNN classifier
def my_grid_search(X, y, parameters, model):

	print("...Calculating the best parameters...")

	# Split the dataset into two equal parts
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


def get_X_and_y(folder):

	folders_names = ['bishop','king','knight','queen','rook']

	X = joblib.load('descriptors_test/'+folder+'/pawn/pawn.pkl')
	num_of_positives = len(X)

	for folder_name in folders_names:
		X = np.concatenate((X,joblib.load('descriptors_test/'+folder+'/'+folder_name+'/'+folder_name+'.pkl')))

	num_of_negatives = len(X) - num_of_positives

	y = np.ones((num_of_positives, 1))
	y = np.ravel(np.concatenate((y,np.zeros((num_of_negatives, 1)))))

	return X,y

def get_X_and_y_pawn_king(folder):

	folders_names = ['king']

	X = joblib.load('descriptors_test/'+folder+'/pawn/pawn.pkl')
	num_of_positives = len(X)

	for folder_name in folders_names:
		X = np.concatenate((X,joblib.load('descriptors_test/'+folder+'/'+folder_name+'/'+folder_name+'.pkl')))


	num_of_negatives = len(X) - num_of_positives

	y = np.ones((num_of_positives, 1))
	y = np.ravel(np.concatenate((y,np.zeros((num_of_negatives, 1)))))

	return X,y

def chess_train(out_dir_desc, train_classes, train_mod="LR_l1"):
		'''

		Args:
				out_dir_desc: path where image descriptors are saved
				train_classes: classes used to train the classifier
				train_mod: (string) Classification model to be used, options are:
									 LR_l1: L1 Logistic Regression
									 LR_l2: L2 Logistic Regression
									 LR_liblinear: liblinear Logistic Regression
									 SVC_linear: linear SVC
									 KNN: KNeighborsClassifier

		Returns:
				classification_model: trained classification model

		'''

		# print("**************** Training classification model")
		# print out_dir_desc
		# print train_classes
		all_desc = []
		all_class_no = []
		for i, class_name in enumerate(train_classes):
				filename = os.path.join(out_dir_desc, class_name, class_name + ".vocab")
				desc = np.float32(np.array(np.loadtxt(filename, delimiter=',')))

				# if i == 0:
				#     class_no = np.zeros((len(desc), 1))
				# # number_of_positive_samples = len(class_no)
				# else:
				#     # desc = random.sample(desc,number_of_positive_samples/2)
				class_no = np.ones((len(desc), 1)) * i

				all_desc = _loop_list(desc, all_desc)
				for d in class_no:
						all_class_no.append(d[0])

		# train, test = train_test_split(all_desc, train_size=0.8, random_state=44)
		# train_i, test_i = train_test_split(all_class_no, train_size=0.8, random_state=44)

		all_desc = np.float32(all_desc)
		all_class_no = np.int8(all_class_no)

		# if train_mod == "LR_l1":
		#     classification_model = LogisticRegression(C=1, penalty='l1')
		# elif train_mod == "LR_l2":
		#     classification_model = LogisticRegression(C=1, penalty='l2')
		# elif train_mod == "LR_liblinear":
		#     classification_model = LogisticRegression(C=1, solver='liblinear')
		# elif train_mod == "SVC_linear":
		#     classification_model = SVC(kernel='linear', C=1, probability=True, random_state=np.random.RandomState(0))
		# elif train_mod == "KNN":
		#     classification_model = KNeighborsClassifier()
		# else:
		#     raise Exception("Unknown classification model")

		# scores = model_selection.cross_val_score(classification_model, all_desc, all_class_no, cv=5)
		# print("Cross-validation results")
		# print(scores)
		# print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
		#
		# classification_model.fit(train, train_i)
		# ans = classification_model.score(test, test_i)
		# print(ans)

		# **************
		# Set the parameters by cross-validation
		if train_mod == "SVM":
				parameters_list = [{'kernel': ['rbf'], 'C': [1, 10, 100, 1000]},
													 {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
		elif train_mod == "LR":
				parameters_list = [{'penalty': ['l2'], 'C': [1, 10, 100, 1000], 'solver': ['lbfgs']},
													 {'penalty': ['l2'], 'C': [1, 10, 100, 1000], 'solver': ['liblinear']},
													 {'penalty': ['l2'], 'C': [1, 10, 100, 1000], 'solver': ['newton-cg']},
													 {'penalty': ['l1'], 'C': [1, 10, 100, 1000], 'solver': ['liblinear']}]
		else:
				raise Exception("Unknown classification model")

		params_opt = grid_search(all_desc, all_class_no, parameters_list, train_mod)
		# **************

		if train_mod == "LR":
				classification_model = LogisticRegression(C=params_opt['C'], penalty=params_opt['penalty'],
																									solver=params_opt['solver'])
		elif train_mod == "SVM":
				classification_model = SVC(C=params_opt['C'], kernel=params_opt['kernel'], probability=True,
																	 random_state=np.random.RandomState(0))
		else:
				raise Exception("Unknown classification model")

		if np.max(all_class_no) >= 2:
				X_train, X_test, y_train, y_test = train_test_split(all_desc, all_class_no, test_size=0.3, random_state=0)
				classification_model.fit(X_train, y_train)
				scores = classification_model.score(X_test, y_test)
				print("Accuracy: %0.2f" % scores.mean())

		elif np.max(all_class_no) < 2:
				cv_kf = StratifiedKFold(n_splits=6)

				mean_tpr = 0.0
				mean_fpr = np.linspace(0, 1, 100)

				from itertools import cycle
				from scipy import interp

				colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange'])
				lw = 2

				i = 0
				for (train, test), color in zip(cv_kf.split(all_desc, all_class_no), colors):
						probas_ = classification_model.fit(all_desc[train], all_class_no[train]).predict_proba(all_desc[test])
						# Compute ROC curve and area the curve
						fpr, tpr, thresholds = roc_curve(all_class_no[test], probas_[:, 1])
						mean_tpr += interp(mean_fpr, fpr, tpr)
						mean_tpr[0] = 0.0
						roc_auc = auc(fpr, tpr)
						plt.plot(fpr, tpr, lw=lw, color=color,
										 label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

						i += 1
				plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k',
								 label='Luck')

				mean_tpr /= cv_kf.get_n_splits(all_desc, all_class_no)
				mean_tpr[-1] = 1.0
				mean_auc = auc(mean_fpr, mean_tpr)
				plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--',
								 label='Mean ROC (area = %0.2f)' % mean_auc, lw=lw)

				plt.xlim([-0.05, 1.05])
				plt.ylim([-0.05, 1.05])
				plt.xlabel('False Positive Rate')
				plt.ylabel('True Positive Rate')
				plt.title('Receiver operating characteristic example')
				plt.legend(loc="lower right")
				plt.show()

		# Compute confusion matrix
		# from sklearn.metrics import confusion_matrix
		# y_pred = classification_model.predict(X_test)
		# cnf_matrix = confusion_matrix(y_test, y_pred)
		# np.set_printoptions(precision=2)

		# Plot non-normalized confusion matrix
		# plt.figure()
		# plot_confusion_matrix(cnf_matrix, classes=train_classes,
		#                       title='Confusion matrix, without normalization')
		#
		# # Plot normalized confusion matrix
		# plt.figure()
		# plot_confusion_matrix(cnf_matrix, classes=train_classes, normalize=True,
		#                       title='Normalized confusion matrix')
		#
		# plt.show()

		print("**************** Done!")

		return classification_model


def _loop_list(in_list, out_list):
		for d in in_list:
				out_list.append(d)

		return out_list


def grid_search(X, y, tuned_parameters, train_mod):
		# Split the dataset in two equal parts
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

		scores = ['precision', 'recall']

		for score in scores:
				print("# Tuning hyper-parameters for %s" % score)
				print ""

				if train_mod == "SVM":
						clf = GridSearchCV(SVC(C=1, probability=True, random_state=np.random.RandomState(0)), tuned_parameters,
															 cv=5, scoring='%s_macro' % score)
				if train_mod == "LR":
						clf = GridSearchCV(LogisticRegression(), tuned_parameters, cv=5, scoring='%s_macro' % score)

				clf.fit(X_train, y_train)

				print("Best parameters set found on development set:")
				print ""
				print(clf.best_params_)
				print ""
				# print("Grid scores on development set:")
				# print ""
				# means = clf.cv_results_['mean_test_score']
				# stds = clf.cv_results_['std_test_score']
				# for mean, std, params in zip(means, stds, clf.cv_results_['params']):
				#     print("%0.3f (+/-%0.03f) for %r"
				#           % (mean, std * 2, params))
				# print ""

				print("Detailed classification report:")
				print ""
				print("The model is trained on the full development set.")
				print("The scores are computed on the full evaluation set.")
				print ""
				y_true, y_pred = y_test, clf.predict(X_test)
				print(classification_report(y_true, y_pred))
				print ""

		return clf.best_params_


def plot_confusion_matrix(cm, classes,
													normalize=False,
													title='Confusion matrix',
													cmap=plt.cm.get_cmap('Blues')):
		"""
		This function prints and plots the confusion matrix.
		Normalization can be applied by setting `normalize=True`.
		"""
		plt.imshow(cm, interpolation='nearest', cmap=cmap)
		plt.title(title)
		plt.colorbar()
		tick_marks = np.arange(len(classes))
		plt.xticks(tick_marks, classes, rotation=45)
		plt.yticks(tick_marks, classes)

		if normalize:
				cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
				print("Normalized confusion matrix")
		else:
				print('Confusion matrix, without normalization')

		print(cm)

		import itertools
		thresh = cm.max() / 2.
		for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
				plt.text(j, i, cm[i, j],
								 horizontalalignment="center",
								 color="white" if cm[i, j] > thresh else "black")

		plt.tight_layout()
		plt.ylabel('True label')
		plt.xlabel('Predicted label')
