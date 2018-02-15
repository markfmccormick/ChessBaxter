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

def create_heatmap(image, stepSize, windowSize, model, heatmap, countmap, path):
	counter = 0

	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			window = image[y:y+windowSize[1], x:x+windowSize[0]]
			if window.shape[1] != windowSize[0] or \
							window.shape[0] != windowSize[1]:
				continue

			#cv2.imwrite(path+str(counter)+".jpg", window)
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
	count = 71
	for points in chess_square_points:
		offset = int(count/8)*5
		square = heatmap[int(points[1][0][0])-offset:int(points[1][1][0]), int(points[0][0][1]):int(points[1][1][1])]
		square_count = countmap[int(points[1][0][0])-offset:int(points[1][1][0]), int(points[0][0][1]):int(points[1][1][1])]
		squares.append(square)
		squares_count.append(square_count)
		count -= 1
	return squares, squares_count

# TODO
# Lots of work and experimentation here to figure out the best way to do this
def label_squares(chess_squares, chess_squares_count):
    # List ordered to match labels.txt file
	label_strings = ["bishop", "king", "knight", "pawn", "queen", "rook",
					"square", "BISHOP", "KING", "KNIGHT", "PAWN", "QUEEN", "ROOK"]
	square_labels = []
	for square in chess_squares:
		high = square.argmax(axis=0)
		print square.shape
		print high
	return square_labels

def label_squares_test(chess_squares, chess_squares_count, heatmap, countmap):
	# List ordered to match labels.txt file
	label_strings = ["bishop", "king", "knight", "pawn", "queen", "rook",
					"square", "BISHOP", "KING", "KNIGHT", "PAWN", "QUEEN", "ROOK"]
	king_or_queen = [1,4,8,11]
	square_labels = []

	for square in chess_squares:
		high = square.argmax(axis=0)
		if high[2] in king_or_queen:
			column = 1

	return square_labels

# model_path = "retrained_graph.pb"
model_path = "models/inception13.pb"
labels_path = "labels.txt"
#labels_path = "inception12.txt"
# labels_path = "inception12.txt"
labels = []
with open(labels_path) as image_labels:
	for line in image_labels:
		line = line.strip('\n')
		line = line.replace(" ", "_")
		labels.append(line)

"""
# Testing baxter loop
square_labels = ["rook","knight","bishop","queen","king","bishop","knight","rook",
		   "pawn","pawn","pawn","pawn","pawn","pawn","pawn","pawn",
	 	   "square","square","square","square","square","square","square","square",
	 	   "square","square","square","square","square","square","square","square",
	 	   "square","square","square","square","square","square","square","square",
	 	   "square","square","square","square","square","square","square","square",
	 	   "PAWN","PAWN","PAWN","PAWN","PAWN","PAWN","PAWN","PAWN",
	 	   "ROOK","KNIGHT","BISHOP","QUEEN","KING","BISHOP","KNIGHT","ROOK"]
board_state_string = create_board_string(square_labels)
board_state_string += " w KQkq - 0 0"
game_over = ""
while game_over == "":
    	moved_board_state_string, game_over, best_move = my_next_move(board_state_string)
	initial_square = best_move.uci()[0:2]
	final_square = best_move.uci()[2:4]
	print "Move made: "+best_move.uci()
	# Baxter performs the move

	board = chess.Board(moved_board_state_string)
	if board.is_checkmate():
    		game_over = "Checkmate, I lost."
	elif board.is_game_over():
		game_over = "Draw"
	else:
		game_over = ""
		
	user_move = raw_input("Enter the move you made: ")
	move = chess.Move.from_uci(user_move)
	board.push(move)
	board_state_string = str(board.fen).split("\'")[1]
	print "Board after user move: "
	print board
"""
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
    	
imgpath = "kinect_images_new/white_front/tall.jpeg"
imgpath = "board_images/camera_image3.jpeg"

chessboard_keypoints = get_keypoints(imgpath)[0]

chess_square_points = create_chess_square_points(chessboard_keypoints)

img = cv2.imread(imgpath)
img = crop_image(chessboard_keypoints, img)
cv2.imshow("Cropped", img)
cv2.waitKey(0)

window_y = 100
window_x = 100
stepSize = 20
# 13 dimensional because there are 13 possible classifications
heatmap = np.zeros((img.shape[0], img.shape[1], 13))
countmap = np.zeros((img.shape[0], img.shape[1]))
model = Model(model_path)
path=""
# for x in range(0, 41, 10):
	# heatmap, countmap = create_heatmap(img, stepSize, (window_x+x, window_y+x), model, heatmap, countmap,path)
heatmap, countmap = create_heatmap(img, stepSize, (window_x, window_y), model, heatmap, countmap, "sliding_window/")

chess_squares, chess_squares_count = create_chess_squares(chess_square_points, heatmap, countmap)

# square_labels = label_squares(chess_squares, chess_squares_count)

# board_state_string = create_board_string(square_labels)
# board_state_string.append(" w KQkq - 0 0")

# moved_board_state_string, game_over, best_move = my_next_move(board_state_string)
# if game_over == "":
#     # Game not over
# 	print "Game not over"

visualise_heatmap(img, heatmap, countmap, labels, "heatmaps/")
"""
# Piece classification testing, checking different angles
angle_test = glob.glob('angle_test/*')
for file in angle_test:
	imgpath=file
	print file
	try:
		os.mkdir("heatmaps/"+file[11:-4])
#    		chessboard_keypoints = get_keypoints(imgpath)[0]
	#	chess_square_points = create_chess_square_points(chessboard_keypoints)
		img = cv2.imread(imgpath)
		img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
	#	img = crop_image(chessboard_keypoints, img)
		heatmap = np.zeros((img.shape[0], img.shape[1], 13))
		countmap = np.zeros((img.shape[0], img.shape[1]))
		model = Model(model_path)
		path = "sliding_window/"+file[11:-4]
		os.mkdir(path)
		for x in range(0, 41, 10):
			heatmap, countmap = create_heatmap(img, stepSize, (window_x+x, window_y+x), model, heatmap, countmap, path)
		#heatmap, countmap = create_heatmap(img, stepSize, (window_x, window_y), model, heatmap, countmap, path)
		visualise_heatmap(img, heatmap, countmap, labels, "heatmaps/"+file[11:-4]+"/")
	except:
		print file
		pass

# Final chess game loop - not in use while still testing
model_path = "models/inception9.pb"
labels_path = "labels.txt"
labels = []
with open(labels_path) as image_labels:
	for line in image_labels:
		line = line.strip('\n')
		line = line.replace(" ", "_")
		labels.append(line)
window_y = 80
window_x = 80
stepSize = 40

result = "not yet"
move = 0
while result == "":
	list_of_files = glob.glob('board_images/*')
	imgpath = max(list_of_files, key=os.path.getctime)

	chessboard_keypoints = get_keypoints(imgpath)[0]
	chess_square_points = create_chess_square_points(chessboard_keypoints)

	img = cv2.imread(imgpath)
	img = crop_image(chessboard_keypoints, img)

	# 13 dimensional because there are 13 possible classifications
	heatmap = np.zeros((img.shape[0], img.shape[1], 13))
	countmap = np.zeros((img.shape[0], img.shape[1]))
	model = Model(model_path)
	for x in range(0, 41, 10):
		heatmap, countmap = create_heatmap(img, stepSize, (window_x+x, window_y+x), model, heatmap, countmap)

	chess_squares, chess_squares_count = create_chess_squares(chess_square_points, heatmap, countmap)
	square_labels = label_squares(chess_squares, chess_squares_count)

	board_state_string = create_board_string(square_labels)

	moved_board_state_string, game_over, best_move = my_next_move(board_state_string)
	if game_over == "":
		# Game not over
		print ""
"""