from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
import numpy as np
from chess_pieces.extra_tools import hog

def black_or_white(image):

	black_white_classifier = joblib.load('chess_pieces/classifiers/colour_hist/black_white_classifier.pkl')

	colour_histogram = hog(image)
	prediction = black_white_classifier.predict(colour_histogram.reshape(1,-1))

	if prediction == 1:
		return "white", black_white_classifier.predict_proba(colour_histogram.reshape(1,-1))
	else:
		return "black", black_white_classifier.predict_proba(colour_histogram.reshape(1,-1))
