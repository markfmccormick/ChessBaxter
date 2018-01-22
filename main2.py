import matplotlib as mpl
mpl.use('GTKAgg')

import glob
import re
import time
import os
import sys

import tensorflow as tf
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# legacy imports to be removed or fixed
from final_sliding_window import final_sliding_window
from label_colour import label_colour
from label_image import label_image
from chess_move import my_next_move
from chessboard_detector import chessboard_homography
from label_square import label_square

from detect_chessboard import get_keypoints

class Model(object):

	def __init__(self, model_path):
		self.session = tf.Session()

		with tf.gfile.FastGFile(model_path, 'rb') as f:
			graph_def = tf.GraphDef()
			graph_def.ParseFromString(f.read())
			_ = tf.import_graph_def(graph_def, name='graph1')

		self.softmax_tensor = self.session.graph.get_tensor_by_name('graph1/final_result:0')

	def predict(self, image_data):
		# image_data = tf.gfile.FastGFile(image_path, 'rb').read()
		# predictions =  self.session.run(self.softmax_tensor, {'graph1/DecodeJpeg/contents:0': image_data})
		predictions = self.session.run(self.softmax_tensor, {'graph1/DecodeJpeg:0': image_data})
		return predictions

# Natural human sorting
def atoi(text):
	return int(text) if text.isdigit() else text

def natural_keys(text):
	return [ atoi(c) for c in re.split('(\d+)', text) ]

pieces = ["rook","knight","bishop","queen","king","pawn"]

returned_state_of_the_board = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 0"
result = ""

def print_prediction(predictions):
	label_lines = [line.rstrip() for line
						in tf.gfile.GFile("retrained_labels.txt")]
	# Sort to show labels of first prediction in order of confidence
	top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

	boo = True
	# print "\n"
	for node_id in top_k:
		human_string = label_lines[node_id]
		score = predictions[0][node_id]
		# print('%s (score = %.5f)' % (human_string, score))
		if boo:
			prediction = human_string
			prediction_score = score
			boo = False

	return prediction, prediction_score

def create_heatmap(image, stepSize, windowSize, model_path):

	heatmap = np.zeros((image.shape[0], image.shape[1], 6))
	countmap = np.zeros((image.shape[0], image.shape[1]))

	model = Model(model_path)

	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			window = image[y:y+windowSize[1], x:x+windowSize[0]]
			if window.shape[1] != windowSize[0] or \
							window.shape[0] != windowSize[1]:
				continue

			window = np.array(window)
			predictions = model.predict(window)

			for n in range(windowSize[1]):
				for m in range(windowSize[0]):
					heatmap[y+n][x+m] += predictions[0]
					countmap[y+n][x+m] += 1

	return heatmap, countmap

def visualise_heatmap(img, heatmap, countmap, savefig):
	pawnmap = np.zeros((img.shape[0], img.shape[1]))

	for x in range(pawnmap.shape[0]):
		for y in range(pawnmap.shape[1]):
			pawnmap[x][y] = heatmap[x][y][2]

	print "Creating heatmap"

	ax = sns.heatmap(pawnmap, cbar = False)
	plt.axis('off')
	if savefig:
		plt.savefig(imgpath[:-5]+"-map.png", bbox_inches='tight')
	plt.show()

def crop_image(points, img):
    # TODO
	# Investigate exact values for these margins later
	left = int(points[72][0]-20)
	right = int(points[80][0]+20)
	top = int(points[0][1]-70)
	bottom = int(points[80][1]+40)

	return img[top:bottom, left:right]

def create_chess_square_points(chessboard_keypoints):
	keypoints = chessboard_keypoints.reshape(9,9,2)
	positions = []
	for y in range(8):
		for x in range(8):
			square = [[],[]]
			square[0].append(keypoints[y][x])
			square[0].append(keypoints[y][x+1])
			square[1].append(keypoints[y+1][x])
			square[1].append(keypoints[y+1][x+1])
			positions.append(square)
	return np.array(positions, dtype="float32")

def create_chess_squares(chess_square_points, heatmap, countmap):
	squares = []
	squares_count = []
	for points in chess_square_points:
		square = heatmap[int(points[1][0][0]):int(points[1][1][0]), int(points[0][0][1]):int(points[1][1][1])]
		square_count = countmap[int(points[1][0][0]):int(points[1][1][0]), int(points[0][0][1]):int(points[1][1][1])]
		squares.append(square)
		squares_count.append(square_count)
	return squares, squares_count

model_path = "retrained_graph.pb"

imgpath = "kinect_images_new/white_front/middle.jpeg"
chessboard_keypoints = get_keypoints(imgpath)[0]

chess_square_points = create_chess_square_points(chessboard_keypoints)

img = cv2.imread(imgpath)
img = crop_image(chessboard_keypoints, img)

window_y = 120
window_x = 90
stepSize = 20
heatmap, countmap = create_heatmap(img, stepSize, (window_x, window_y), model_path)

chess_squares, chess_squares_count = create_chess_squares(chess_square_points, heatmap, countmap)

savefig = False
visualise_heatmap(img, heatmap, countmap, savefig)
