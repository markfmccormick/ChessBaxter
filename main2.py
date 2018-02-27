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
		for y in range(int(point[0])-5, int(point[0])+5):
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

		# Show cropped image, for debugging purposes
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
	heatmap, countmap = create_heatmap(img, stepSize, (window_x, window_y), model, heatmap, countmap, 0.15)
	visualise_heatmap(img, heatmap, countmap, labels, "heatmaps/")

	return heatmap, countmap, center_keypoints, crop_points

def box_total_data(center_keypoints, heatmap, countmap, labels, labels_map):
	square_data = []
	for point in center_keypoints:
		totals = [0,0,0,0,0,0,0,0,0,0,0,0,0]
		# totals = [0,0,0,0,0,0,0]
		for y in range(int(point[0])-5, int(point[0])+5):
			for x in range(int(point[1])-5, int(point[1])+5):
				totals += heatmap[x][y]
		square_data.append(totals)
	return square_data

def square_classification_smart_precedence(square_data, labels, labels_map, piece_count_master):
	piece_count = piece_count_master
	square_labels = ["" for i in square_data]
	blank = [0,0,0,0,0,0,0,0,0,0,0,0,0]
	precedence_list = ["empty_square", "black_king", "black_queen", "white_king", "white_queen", "black_rook", "white_rook",
						"white_pawn", "black_pawn", "white_bishop", "white_knight", "black_knight", "black_bishop"]

	for piece in precedence_list:
		piece_index = labels.index(piece)
		for count in range(piece_count[piece]):
			max_index = -1
			max_value = -1
			for i in range(len(square_data)):
				if square_data[i][piece_index] > max_value:
					max_index = i
					max_value = square_data[i][piece_index]
			square_labels[max_index] = labels_map[labels[piece_index]]
			square_data[max_index] = blank
		for j in range(len(square_data)):
			square_data[j][piece_index] = 0

	return square_labels

model_path = "models/inception14.pb"
model_path = "models/inception22.pb"
labels_path = "labels.txt"
# labels_path = "inception17.txt"
imgpath = "test_images/camera_image2.jpeg"
imgpath = "pictures/1.jpeg"
# imgpath = "pictures/queen/1.jpeg"

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
# For testing
# piece_count = {"black_pawn": 0, "black_knight": 0, "black_bishop": 0, "black_king": 0, "black_queen": 1, "black_rook": 0,
# 			"empty_square": 60, "white_pawn": 0, "white_knight": 0, "white_bishop": 0, "white_king": 0, "white_queen": 1, "white_rook": 0}

board_square_map= {"a1":0 ,"a2":1, "a3":2, "a4":3, "a5":4, "a6":5, "a7":6, "a8":7, 
					"b1":8 ,"b2":9, "b3":10, "b4":11, "b5":12, "b6":13, "b7":14, "b8":15,
					"c1":16 ,"c2":17, "c3":18, "c4":19, "c5":20, "c6":21, "c7":22, "c8":23, 
					"d1":24 ,"d2":25, "d3":26, "d4":27, "d5":28, "d6":29, "d7":30, "d8":31, 
					"e1":32 ,"e2":33, "e3":34, "e4":35, "e5":36, "e6":37, "e7":38, "e8":39, 
					"f1":40 ,"f2":41, "f3":42, "f4":43, "f5":44, "f6":45, "f7":46, "f8":47, 
					"g1":48 ,"g2":49, "g3":50, "g4":51, "g5":52, "g6":53, "g7":54, "g8":55, 
					"h1":56 ,"h2":57, "h3":58, "h4":59, "h5":60, "h6":61, "h7":62, "h8":63}
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
square_labels = square_classification_smart_precedence(square_data, labels, labels_map, piece_count)
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