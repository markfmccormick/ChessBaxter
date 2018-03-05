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
from naive_classification import show_naive_classification
from load_maps import create_constants
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
	heatmap, countmap = create_heatmap(img, stepSize, (window_x, window_y), model, heatmap, countmap, 0.0)

	# Potential thresholding technique
	for y in range(len(heatmap)):
		for x in range(len(heatmap[y])):
			for z in range(len(heatmap[y][x])):
				if heatmap[y][x][z] <= 1.5:
					heatmap[y][x][z] = 0

	# visualise_heatmap(img, heatmap, countmap, labels, "heatmaps/")

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

def square_classification_smart_precedence(square_data, labels, labels_map, piece_count_master, precedence_list):
	piece_count = piece_count_master
	square_labels = ["" for i in square_data]
	blank = [0,0,0,0,0,0,0,0,0,0,0,0,0]

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

def scaling_debug(square_data):
	test = []
	for i in range(len(square_data)):
		test.append(int(square_data[i][5]))
	print np.reshape(test, (8,8))

def apply_scaling(board, square_data, labels, labels_map, board_square_map):
	
	# scaling_debug(square_data)

	square_data_ordered = np.reshape(square_data, (8,8,13))
	square_data_ordered = np.reshape(square_data_ordered, (8,104))
	square_data_ordered = np.flipud(square_data_ordered)
	square_data_ordered = np.reshape(square_data_ordered, (8,8,13))

	square_data_ordered = np.reshape(square_data_ordered, (64,13))

	# scaling_debug(square_data)
	
	for i in range(len(square_data_ordered)):
		if board.piece_at(i) != None:
			piece = letter_count_map[str(board.piece_at(i))]
		else:
			piece = "empty_square"
		square = board_square_map.keys()[board_square_map.values().index(i)]

		square_data_ordered[i][labels.index(piece)] *= 10

		for move in board.legal_moves:
			move = move.uci()
			if move[0:2] == square:
				index = board_square_map[move[2:4]]
				square_data_ordered[index][labels.index(piece)] *= 5

	square_data_ordered = np.reshape(square_data_ordered, (8,8,13))
	square_data_ordered = np.reshape(square_data_ordered, (8,104))
	square_data_ordered = np.flipud(square_data_ordered)
	square_data_ordered = np.reshape(square_data_ordered, (8,8,13))

	square_data = np.reshape(square_data_ordered, (64,13))

	# scaling_debug(square_data)

	return square_data
	
# Gets the move made by the user
# Assumes perfect board state detection, which I don't have
def get_move_made(pre_board, post_board, board_square_map, piece_count, castling_rights, player_colour, letter_count_map):
	# Can be True if a piece is taken
	piece_taken = False

	# Check for castling move
	if player_colour == "white":
		if castling_rights == "QUEENSIDE":
			if str(post_board.piece_at(board_square_map["c1"])) == "K":
				return "e1c1", piece_count, piece_taken
		elif castling_rights == "KINGSIDE":
			if str(post_board.piece_at(board_square_map["g1"])) == "K":
				return "e1g1", piece_count, piece_taken
		elif castling_rights == "BOTH":
			if str(post_board.piece_at(board_square_map["c1"])) == "K":
				return "e1c1", piece_count, piece_taken
			elif str(post_board.piece_at(board_square_map["g1"])) == "K":
				return "e1g1", piece_count, piece_taken
	elif player_colour == "black":
		if castling_rights == "QUEENSIDE":
			if str(post_board.piece_at(board_square_map["c8"])) == "k":
				return "e8c8", piece_count, piece_taken
		elif castling_rights == "KINGSIDE":
			if str(post_board.piece_at(board_square_map["g8"])) == "k":
				return "e8g8", piece_count, piece_taken
		elif castling_rights == "BOTH":
			if str(post_board.piece_at(board_square_map["c8"])) == "k":
				return "e8c8", piece_count, piece_taken
			elif str(post_board.piece_at(board_square_map["g8"])) == "k":
				return "e8g8", piece_count, piece_taken

	for move in pre_board.legal_moves:
		move = move.uci()
		if pre_board.piece_at(board_square_map[move[0:2]]) == post_board.piece_at(board_square_map[move[2:4]]):
			if str(pre_board.piece_at(board_square_map[move[2:4]])) != 'None':
				piece_count[letter_count_map[str(pre_board.piece_at(board_square_map[move[2:4]]))]] -= 1
				move_made = move
				piece_taken = True
				break
			else:
				move_made = move
				break

	return move_made, piece_count, piece_taken


labels, labels_map, piece_count, board_square_map, position_map, base_right, base_left, precedence_list, letter_count_map = create_constants()
# model_path = "models/inception14.pb"
model_path = "models/inception22.pb"
labels_path = "labels.txt"
# labels_path = "inception17.txt"
# imgpath = "test_images/camera_image2.jpeg"
imgpath = "pictures/3.jpeg"
# imgpath = "pictures/queen/1.jpeg"

# list_of_files = glob.glob('board_images/*')
# imgpath = max(list_of_files, key=os.path.getctime)

model = Model(model_path)

window_y = 100
window_x = 100
stepSize = 20

heatmap, countmap, center_keypoints, crop_points = classify_board(imgpath)

# for y in range(len(heatmap)):
# 	for x in range(len(heatmap[y])):
# 		heatmap[y][x] = heatmap[y][x]/countmap[y][x]

show_naive_classification(center_keypoints, heatmap, countmap, labels, labels_map)

square_data = box_total_data(center_keypoints, heatmap, countmap, labels, labels_map)
square_data = apply_scaling(chess.Board(), square_data, labels, labels_map, board_square_map)
square_labels = square_classification_smart_precedence(square_data, labels, labels_map, piece_count, precedence_list)
board_state_string = create_board_string(square_labels)
board_state_string += " w KQkq - 0 1"
board = chess.Board(board_state_string)
print "Method - Box total smart: "
print board
"""
rospy.init_node('Chess_Baxter')
right = baxter_interface.Limb('right')
left = baxter_interface.Limb('left')
right.set_joint_position_speed(0.8)
left.set_joint_position_speed(0.8)
right.move_to_joint_positions(base_right)
left.move_to_joint_positions(base_left)
right_gripper = baxter_interface.Gripper('right')
right_gripper.calibrate()
right_gripper.set_parameters({"velocity":50.0, 
						"moving_force":20.0, 
						"holding_force":10.0,
						"dead_zone":5.0})

if len(sys.argv) != 1:
	print "Usage: python main2.py [baxter_colour]"
	print "colour is 'black' or 'white', corresponds to the colour Baxter is playing as"
	print "The game must be played from the start, or after your first move if Baxter is black"
	sys.exit()

baxter_colour = sys.argv[1]
if baxter_colour == "white":
	move = 0
	player_colour = "black"
elif baxter_colour == "black":
	wait = raw_input("Make your move then press Enter: ")
	move = 1
	player_colour = "white"
else:
	print "Usage: python main2.py [baxter_colour]"
	print "colour is 'black' or 'white', corresponds to the colour Baxter is playing as"
	print "The game must be played from the start, or after your first move if Baxter is black"
	sys.exit()

list_of_files = glob.glob('board_images/*')
imgpath = max(list_of_files, key=os.path.getctime)

heatmap, countmap, center_keypoints, crop_points = classify_board(imgpath)
square_data = box_total_data(center_keypoints, heatmap, countmap, labels, labels_map)
square_labels = square_classification_smart_precedence(square_data, labels, labels_map, piece_count, precedence_list)
board_state_string = create_board_string(square_labels)
board_state_string += "b KQkq - 0 1"
pre_board = chess.Board(board_state_string)
print "Board before Baxter move: "
print pre_board

board = pre_board

game_over = "test"
while game_over == "":

	moved_board_state_string, game_over, best_move = my_next_move(board_state_string)
	post_board = chess.Board(moved_board_state_string)
	castling = post_board.is_castling(best_move)

	initial = best_move.uci()[0:2]
	final = best_move.uci()[2:4]
	print initial_square, final_square
	print "Move made: "+best_move.uci()
	print "Board after Baxter move: "
	print post_board

	board = post_board

	# Perform move with Baxter
	if castling == False:
    	pivot = ""
    	capture = False
		if board.piece_at(board_square_map[final]) != None:
    		capture = True
		if initial not in pivot_points and final not in pivot_points:
    		pivot = "None"
    	if initial in pivot_points and final in pivot_points:
    		pivot = "None"
		elif initial in pivot_points and final not in pivot_points:	
			pivot = "To"
		elif initial not in pivot_points and final in pivot_points:
    		pivot = "From"
		perform_move(initial, final, position_map, right, left, gripper, pivot, capture)
	else:
    	capture = False
		pivot = "None"
		if final in pivot_points:
    		perform_move(final, "pivot_from", position_map, right, left, gripper, pivot, capture)
			perform_move(initial, final, position_map, right, left, gripper, pivot, capture)
			perform_move("pivot_from", initial, position_map, right, left, gripper, pivot, capture)
		else:
    		perform_move(final, "pivot_to", position_map, right, left, gripper, pivot, capture)
			perform_move(initial, final, position_map, right, left, gripper, pivot, capture)
			perform_move("pivot_to", initial, position_map, right, left, gripper, pivot, capture)

	if post_board.is_checkmate():
			game_over = "Checkmate, I lost."
	elif post_board.is_game_over():
		game_over = "Draw"
	else:
		game_over = ""

	wait = raw_input("Make your move then press Enter: ")

	move_made = ""
	while move_made == "":
		list_of_files = glob.glob('board_images/*')
		imgpath = max(list_of_files, key=os.path.getctime)

		heatmap, countmap, center_keypoints, crop_points = classify_board(imgpath)
		square_data = box_total_data(center_keypoints, heatmap, countmap, labels, labels_map)
		square_labels = square_classification_smart_precedence(square_data, labels, labels_map, piece_count, precedence_list)
		board_state_string = create_board_string(square_labels)
		if baxter_colour == "white":
			board_state_string += "w KQkq - 0 1"
		else:
			board_state_string += "b KQkq - 0 1"
		pre_board = chess.Board(board_state_string)
		print "Board before Baxter move: "
		print pre_board

		if baxter_colour == "white":
			queenside = post_board.has_queenside_castling_rights(chess.BLACK)
			kingside = post_board.has_kingside_castling_rights(chess.BLACK)
		else:
			queenside = post_board.has_queenside_castling_rights(chess.WHITE)
			kingside = post_board.has_kingside_castling_rights(chess.WHITE)
		
		castling_rights = ""
		if queenside and kingside:
			castling_rights = "BOTH"
		elif queenside:
			castling_rights = "QUEENSIDE"
		elif kingside:
			castling_rights = "KINGSIDE"

		move_made, piece_count, piece_taken = get_move_made(post_board, pre_board, board_square_map, piece_count, castling_rights, player_colour, letter_count_map)
		if move_made == "":
			print "Board state detection error, trying again with lower step_size"
			stepSize -= 5
			if stepSize == 0:
				print "Cannot get board state, exiting"
				sys.exit()
			continue
		if piece_taken:
			heatmap, countmap, center_keypoints, crop_points = classify_board(imgpath)
			square_data = box_total_data(center_keypoints, heatmap, countmap, labels, labels_map)
			square_labels = square_classification_smart_precedence(square_data, labels, labels_map, piece_count, precedence_list)
			board_state_string = create_board_string(square_labels)
			if baxter_colour == "white":
				board_state_string += "w KQkq - 0 1"
			else:
				board_state_string += "b KQkq - 0 1"
			pre_board = chess.Board(board_state_string)

			move_made, piece_count, piece_taken = get_move_made(post_board, pre_board, board_square_map, piece_count, castling_rights, player_colour, letter_count_map)
			if move_made == "":
				print "Board state detection error, trying again with lower step_size"
				stepSize -= 5
				if stepSize == 0:
					print "Cannot get board state, exiting"
					sys.exit()
				continue
				
		stepSize = 20
		board.push(move_made)
		board_state_string = str(board.fen).split("\'")[1]
"""