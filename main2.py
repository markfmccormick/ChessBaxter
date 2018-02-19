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
import chess
import chess.uci
import stockfish

from detect_chessboard import get_keypoints
from create_board_string import create_board_string
from chess_move import my_next_move
from move_baxter import perform_move

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

def create_heatmap(image, stepSize, windowSize, model, heatmap, countmap):
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

def get_crop_points(chessboard_keypoints):
    crop_points = {}
	crop_points["top"] = int(chessboard_keypoints[np.argsort(chessboard_keypoints[:, 1])][80][1]+20)
	crop_points["bottom"] = int(chessboard_keypoints[np.argsort(chessboard_keypoints[:, 1])][0][1]-40)
	crop_points["right"] = int(chessboard_keypoints[np.argsort(chessboard_keypoints[:, 0])][80][0]+20)
	crop_points["left"] = int(chessboard_keypoints[np.argsort(chessboard_keypoints[:, 0])][0][0]-20)
	return crop_points

def label_squares_point(center_keypoints, heatmap, countmap, labels, labels_map):
	square_labels = []
	for point in center_keypoints:
		predictions = heatmap[int(point[1])][int(point[0])]
		index = np.argmax(predictions)
		square_labels.append(labels_map[labels[index]])
	return square_labels

def label_squares_box_total(center_keypoints, heatmap, countmap, labels, labels_map):
	square_labels = []
	for point in center_keypoints:
		totals = [0,0,0,0,0,0,0,0,0,0,0,0,0]
		for y in range(int(point[0])-20, int(point[0])+20):
			for x in range(int(point[1])-20, int(point[1])+20):
				totals += heatmap[x][y]
		index = np.argmax(totals)
		square_labels.append(labels_map[labels[index]])
	return square_labels

def label_squares_experimental(center_keypoints, heatmap, countmap, labels, labels_map):
	square_labels = []
	for point in center_keypoints:
		totals = [0,0,0,0,0,0,0,0,0,0,0,0,0]
		for y in range(int(point[0])-20, int(point[0])+20):
			for x in range(int(point[1])-20, int(point[1])+20):
				totals += heatmap[x][y]
		index = np.argmax(totals)
		square_labels.append(labels_map[labels[index]])
	return square_labels

def classify_board():
    # 13 dimensional because there are 13 possible classifications
	heatmap = np.zeros((img.shape[0], img.shape[1], 13))
	countmap = np.zeros((img.shape[0], img.shape[1]))

    corner_keypoints, center_keypoints = get_keypoints(imgpath)
	corner_keypoints = corner_keypoints[0]
	center_keypoints = center_keypoints[0]
	crop_points = get_crop_points(corner_keypoints)

	img = cv2.imread(imgpath)
	img = img[crop_points["bottom"]:crop_points["top"],crop_points["left"]:crop_points["right"]]
	imgpath = "cropped_image.jpeg"
	cv2.imwrite(imgpath, img)
	img = cv2.imread(imgpath)
	corner_keypoints, center_keypoints = get_keypoints(imgpath)
	corner_keypoints = corner_keypoints[0]
	center_keypoints = center_keypoints[0]

	# window_y = 80
	# window_x = 80
	# stepSize = 40
	# for x in range(0, 41, 10):
		# heatmap, countmap = create_heatmap(img, stepSize, (window_x+x, window_y+x), model, heatmap, countmap,path)
	heatmap, countmap = create_heatmap(img, stepSize, (window_x, window_y), model, heatmap, countmap)
	# visualise_heatmap(img, heatmap, countmap, labels, "heatmaps/")

	return heatmap, countmap

model_path = "models/inception14.pb"
labels_path = "labels.txt"
imgpath = "board_images/camera_image2.jpeg"

model = Model(model_path)

labels = []
with open(labels_path) as image_labels:
	for line in image_labels:
		line = line.strip('\n')
		line = line.replace(" ", "_")
		labels.append(line)
labels_map = {"black_pawn": "pawn", "black_knight": "knight", "black_bishop": "bishop", "black_king": "king", "black_queen": "queen", "black_rook": "rook",
			"empty_square": "square", "white_pawn": "PAWN", "white_knight": "KNIGHT", "white_bishop": "BISHOP", "white_king": "KING", "white_queen": "QUEEN", "white_rook": "ROOK"}

position_map = {}
right_joint_labels = ['right_s0', 'right_s1',
				'right_e0', 'right_e1'
				'right_w0', 'right_w1', 'right_w2']
with open("square_positions.txt") as position_labels:
	for line in position_labels:
		square_positions = {}
		joint_positions1 = {}
		joint_positions2 = {}
		if len(line) > 18:
			line = line.strip('\n')
			line = line.split(":")
			square = line[0]
			positions = line[1].split(";")
			values = positions[1].split(",")
			for i in range(len(right_joint_labels)):
				joint_positions1[right_joint_labels[i]] = float(values[i])
			square_positions[positions[0]] = joint_positions1
			values = positions[3].split(",")
			for i in range(len(right_joint_labels)):
				joint_positions2[right_joint_labels[i]] = float(values[i])
			square_positions[positions[2]] = joint_positions2
			position_map[square] = square_positions

window_y = 100
window_x = 100
stepSize = 20

heatmap, countmap = classify_board()

square_labels = label_squares_point(center_keypoints, heatmap, countmap, labels, labels_map)
board_state_string = create_board_string(square_labels)
board_state_string += " w KQkq - 0 0"
board = chess.Board(board_state_string)
print "Method 1 - Center point: "
print board

square_labels = label_squares_box_total(center_keypoints, heatmap, countmap, labels, labels_map)
board_state_string = create_board_string(square_labels)
board_state_string += " w KQkq - 0 0"
board = chess.Board(board_state_string)
print "Method 2 - Box total: "
print board

game_over = "not yet"
move = 0
while game_over == "":
	list_of_files = glob.glob('board_images/*')
	imgpath = max(list_of_files, key=os.path.getctime)

	heatmap, countmap = classify_board()

	square_labels = label_squares_point(center_keypoints, heatmap, countmap, labels, labels_map)
	board_state_string = create_board_string(square_labels)
	if move == 0:
    	board_state_string += " w KQkq - 0 0"

	moved_board_state_string, game_over, best_move = my_next_move(board_state_string)
	initial_square = best_move.uci()[0:2]
	final_square = best_move.uci()[2:4]
	print "Move made: "+best_move.uci()

	perform_move(initial_square, final_square, position_map)

	board = chess.Board(moved_board_state_string)
	if board.is_checkmate():
		game_over = "Checkmate, I lost."
	elif board.is_game_over():
		game_over = "Draw"
	else:
		game_over = ""
