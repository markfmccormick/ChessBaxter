import matplotlib as mpl
# To get mpl working over ssh, can be disabled otherwise or if causing error
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

from detect_chessboard import get_keypoints
#from create_board_string import create_board_string
#from chess_move import my_next_move

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
    # 13 dimensional because there are 13 possible classifications
	heatmap = np.zeros((image.shape[0], image.shape[1], 13))
	countmap = np.zeros((image.shape[0], image.shape[1]))

	model = Model(model_path)

	counter = 0

	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			window = image[y:y+windowSize[1], x:x+windowSize[0]]
			if window.shape[1] != windowSize[0] or \
							window.shape[0] != windowSize[1]:
				continue

			cv2.imwrite("sliding_window/"+str(counter)+".jpg", window)
			counter+=1

			window = np.array(window)
			predictions = model.predict(window)

			for n in range(windowSize[1]):
				for m in range(windowSize[0]):
					heatmap[y+n][x+m] += predictions[0]
					countmap[y+n][x+m] += 1

	return heatmap, countmap

def visualise_heatmap(img, heatmap, countmap, labels, base_path):
	print "Visualising heatmap"

	for piece in range(len(labels)):

		map = np.zeros((img.shape[0], img.shape[1]))
		for x in range(map.shape[0]):
			for y in range(map.shape[1]):
				map[x][y] = heatmap[x][y][piece]

		ax = sns.heatmap(map, cbar = False)
		plt.axis('off')
		plt.savefig(base_path+labels[piece]+".png", bbox_inches='tight')
		# plt.show()
		plt.clf()

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

# TODO
# Lots of work and experimentation here to figure out the best way to do this
def label_squares(chess_squares, chess_squares_count):
    # List ordered to match labels.txt file
	label_strings = ["bishop", "king", "knight", "pawn", "queen", "rook",
					"square", "BISHOP", "KING", "KNIGHT", "PAWN", "QUEEN", "ROOK"]
	square_labels = []
	print np.shape(chess_squares[1])
	return square_labels

# model_path = "retrained_graph.pb"
model_path = "inception4.pb"
labels_path = "inception.txt"
labels = []
with open(labels_path) as image_labels:
	for line in image_labels:
		line = line.strip('\n')
		line = line.replace(" ", "_")
		labels.append(line)

imgpath = "kinect_images_new/white_front/middle.jpeg"

chessboard_keypoints = get_keypoints(imgpath)[0]

chess_square_points = create_chess_square_points(chessboard_keypoints)

img = cv2.imread(imgpath)
img = crop_image(chessboard_keypoints, img)

window_y = 100
window_x = 100
stepSize = 30
heatmap, countmap = create_heatmap(img, stepSize, (window_x, window_y), model_path)

chess_squares, chess_squares_count = create_chess_squares(chess_square_points, heatmap, countmap)
square_labels = label_squares(chess_squares, chess_squares_count)
"""
board_state_string = create_board_string(square_labels)

moved_board_state_string, game_over = my_next_move(board_state_string)
if game_over == "":
    # Game not over
	print "Game not over"
"""
visualise_heatmap(img, heatmap, countmap, labels, "heatmaps/")
