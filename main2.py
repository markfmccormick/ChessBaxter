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
#from move_baxter import perform_move

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

def create_heatmap(image, stepSize, windowSize, model, heatmap, countmap, threshold):
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
					countmap[y+n][x+m] += 1
					if threshold > 0.0:
						for i in range(len(predictions[0])):
							if predictions[0][i] >= threshold:
								heatmap[y+n][x+m][i] += predictions[0][i]
					else:
						heatmap[y+n][x+m] += predictions[0]
			
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
		# totals = [0,0,0,0,0,0,0]
		for y in range(int(point[0])-10, int(point[0])+10):
			for x in range(int(point[1])-5, int(point[1])+5):
				totals += heatmap[x][y]
		index = np.argmax(totals)
		square_labels.append(labels_map[labels[index]])
	return square_labels

def label_squares_peak(center_keypoints, heatmap, countmap, labels, labels_map):
	square_labels = []
	for point in center_keypoints:
		totals = [0,0,0,0,0,0,0,0,0,0,0,0,0]
		# totals = [0,0,0,0,0,0,0]
		for y in range(int(point[0])-20, int(point[0])+20):
			for x in range(int(point[1])-20, int(point[1])+20):
				for z in range(len(heatmap[x][y])):
                                    if heatmap[x][y][z] > totals[z]:
                                        totals[z] = heatmap[x][y][z]
		index = np.argmax(totals)
		square_labels.append(labels_map[labels[index]])
	return square_labels

def label_squares_center_weighted(center_keypoints, heatmap, countmap, labels, labels_map):
	square_labels = []
	for point in center_keypoints:
		totals = [0,0,0,0,0,0,0,0,0,0,0,0,0]
		# totals = [0,0,0,0,0,0,0]
		county = 1
		for y in range(int(point[0])-20, int(point[0])+20):
                        countx = 1
			for x in range(int(point[1])-20, int(point[1])+20):
				totals += heatmap[x][y]*(countx*county)
                                if x - int(point[1]) > 0:
                                    countx += 1
                                else:
                                    countx -= 1
                        if y - int(point[0]) > 0:
                            county += 1
                        else:
                            county -= 1
		index = np.argmax(totals)
		square_labels.append(labels_map[labels[index]])
	return square_labels

def label_squares_box_total_threshold(center_keypoints, heatmap, countmap, labels, labels_map):
	square_labels = []
	for point in center_keypoints:
		totals = [0,0,0,0,0,0,0,0,0,0,0,0,0]
		# totals = [0,0,0,0,0,0,0]
		for y in range(int(point[0])-20, int(point[0])+20):
			for x in range(int(point[1])-20, int(point[1])+20):
				for z in range(len(heatmap[x][y])):
					if heatmap[x][y][z] > 0.2:
						totals[z] += heatmap[x][y][z]
		index = np.argmax(totals)
		square_labels.append(labels_map[labels[index]])
	return square_labels

def classify_board(imgpath):
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

    	# cv2.imshow("Cropped image", img)
    	# cv2.waitKey(0)

    # 13 dimensional because there are 13 possible classifications
	# heatmap = np.zeros((img.shape[0], img.shape[1], 7))
	heatmap = np.zeros((img.shape[0], img.shape[1], 13))
	countmap = np.zeros((img.shape[0], img.shape[1]))

	# window_y = 80
	# window_x = 80
	# stepSize = 40
	# for x in range(0, 41, 10):
		# heatmap, countmap = create_heatmap(img, stepSize, (window_x+x, window_y+x), model, heatmap, countmap,path)
	heatmap, countmap = create_heatmap(img, stepSize, (window_x, window_y), model, heatmap, countmap, 0.0)
	visualise_heatmap(img, heatmap, countmap, labels, "heatmaps/")

	return heatmap, countmap, center_keypoints, crop_points

def box_total_data(center_keypoints, heatmap, countmap, labels, labels_map):
	square_data = []
	for point in center_keypoints:
		totals = [0,0,0,0,0,0,0,0,0,0,0,0,0]
		# totals = [0,0,0,0,0,0,0]
		for y in range(int(point[0])-10, int(point[0])+10):
			for x in range(int(point[1])-5, int(point[1])+5):
				totals += heatmap[x][y]
		square_data.append(totals)
	return square_data

def square_classification_smart(square_data, labels, labels_map, piece_count):
	pass

model_path = "models/inception14.pb"
model_path = "models/inception20.pb"
labels_path = "labels.txt"
# labels_path = "inception17.txt"
imgpath = "test_images/camera_image2.jpeg"
imgpath = "pictures/2.jpeg"

model = Model(model_path)

labels = []
with open(labels_path) as image_labels:
	for line in image_labels:
		line = line.strip('\n')
		line = line.replace(" ", "_")
		labels.append(line)
labels_map = {"black_pawn": "pawn", "black_knight": "knight", "black_bishop": "bishop", "black_king": "king", "black_queen": "queen", "black_rook": "rook",
			"empty_square": "square", "white_pawn": "PAWN", "white_knight": "KNIGHT", "white_bishop": "BISHOP", "white_king": "KING", "white_queen": "QUEEN", "white_rook": "ROOK"}
piece_count = {"black_pawn": 8, "black_knight": 2, "black_bishop": 2, "black_king": 1, "black_queen": 1, "black_rook": 2,
			"empty_square": 32, "white_pawn": 8, "white_knight": 2, "white_bishop": 2, "white_king": 1, "white_queen": 1, "white_rook": 2}
# labels_map = {"pawn": "pawn", "knight": "knight", "bishop": "bishop", "king": "king", "queen": "queen", "rook": "rook", "empty_square": "square"}

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

heatmap, countmap, center_keypoints, crop_points = classify_board(imgpath)

# for y in range(len(heatmap)):
# 	for x in range(len(heatmap[y])):
# 		heatmap[y][x] = heatmap[y][x]/countmap[y][x]

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

square_labels = label_squares_peak(center_keypoints, heatmap, countmap, labels, labels_map)
board_state_string = create_board_string(square_labels)
board_state_string += " w KQkq - 0 0"
board = chess.Board(board_state_string)
print "Method 3 - Peak value: "
print board

square_labels = label_squares_center_weighted(center_keypoints, heatmap, countmap, labels, labels_map)
board_state_string = create_board_string(square_labels)
board_state_string += " w KQkq - 0 0"
board = chess.Board(board_state_string)
print "Method 4 - Center weighted: "
print board

square_labels = label_squares_box_total_threshold(center_keypoints, heatmap, countmap, labels, labels_map)
board_state_string = create_board_string(square_labels)
board_state_string += " w KQkq - 0 0"
board = chess.Board(board_state_string)
print "Method 5 - Box total threshold: "
print board

img = cv2.imread(imgpath)
img = img[crop_points["bottom"]:crop_points["top"],crop_points["left"]:crop_points["right"]]
imgpath = "cropped_image.jpeg"
cv2.imwrite(imgpath, img)
img = cv2.imread(imgpath)
square_labels = []
for point in center_keypoints:
	window = img[int(point[1])-50:int(point[1])+50, int(point[0])-50:int(point[0])+50]
	predictions = model.predict(window)
	index = np.argmax(predictions)
	square_labels.append(labels_map[labels[index]])
board_state_string = create_board_string(square_labels)
board_state_string += " w KQkq - 0 0"
board = chess.Board(board_state_string)
print "Method 6 - Window around center: "
print board

square_data = box_total_data(center_keypoints, heatmap, countmap, labels, labels_map)
square_labels = square_classification_smart(square_data, labels, labels_map, piece_count)
board_state_string = create_board_string(square_labels)
board_state_string += " w KQkq - 0 0"
board = chess.Board(board_state_string)
print "Method 7 - Box total smart: "
print board

game_over = "test"
move = 0
while game_over == "":
	list_of_files = glob.glob('board_images/*')
	imgpath = max(list_of_files, key=os.path.getctime)

	# heatmap, countmap = classify_board(imgpath)

	# square_labels = label_squares_point(center_keypoints, heatmap, countmap, labels, labels_map)
	# board_state_string = create_board_string(square_labels)
	
	# if move == 0:
    # 	board_state_string += " w KQkq - 0 0"

	# moved_board_state_string, game_over, best_move = my_next_move(board_state_string)
	# initial_square = best_move.uci()[0:2]
	# final_square = best_move.uci()[2:4]
	# print "Move made: "+best_move.uci()

	# #perform_move(initial_square, final_square, position_map)

	# board = chess.Board(moved_board_state_string)
	# if board.is_checkmate():
	# 	game_over = "Checkmate, I lost."
	# elif board.is_game_over():
	# 	game_over = "Draw"
	# else:
	# 	game_over = ""
	heatmap, countmap, center_keypoints = classify_board(imgpath)

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

	square_labels = label_squares_peak(center_keypoints, heatmap, countmap, labels, labels_map)
	board_state_string = create_board_string(square_labels)
	board_state_string += " w KQkq - 0 0"
	board = chess.Board(board_state_string)
	print "Method 3 - Peak value: "
	print board

	square_labels = label_squares_center_weighted(center_keypoints, heatmap, countmap, labels, labels_map)
	board_state_string = create_board_string(square_labels)
	board_state_string += " w KQkq - 0 0"
	board = chess.Board(board_state_string)
	print "Method 4 - Center weighted: "
	print board

	wait = raw_input("Press Enter to continue: ")
