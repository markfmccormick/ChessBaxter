import matplotlib as mpl
mpl.use('GTKAgg')

import glob
import re
import time

import tensorflow as tf
from final_sliding_window import final_sliding_window
from label_colour import label_colour
from label_image import label_image

from chess_move import my_next_move
from chessboard_detector import chessboard_homography
from label_square import label_square

import cv2
import numpy as np
import os, sys
import seaborn as sns
# import matplotlib as mpl
# mpl.use('GTKAgg')
import matplotlib.pyplot as plt



# Natural human sorting
def atoi(text):
	return int(text) if text.isdigit() else text

def natural_keys(text):
	return [ atoi(c) for c in re.split('(\d+)', text) ]

pieces = ["rook","knight","bishop","queen","king","pawn"]

returned_state_of_the_board = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 0"
result = ""

def label_image(image_path):

	# Read in the image_data
	image_data = tf.gfile.FastGFile(image_path, 'rb').read()
	# image_data = image_path

	# Loads label file, strips off carriage return
	label_lines = [line.rstrip() for line
				   in tf.gfile.GFile("retrained_labels.txt")]

	with tf.Session() as sess:

		# Feed the image_data as input to the graph and get first prediction
		softmax_tensor = sess.graph.get_tensor_by_name('graph1/final_result:0')

		predictions = sess.run(softmax_tensor, \
							   {'graph1/DecodeJpeg/contents:0': image_data})
	#
	# 	# Sort to show labels of first prediction in order of confidence
	# 	top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
	#
	# 	boo = True
	# 	# print "\n"
	# 	for node_id in top_k:
	# 		human_string = label_lines[node_id]
	# 		score = predictions[0][node_id]
	# 		# print('%s (score = %.5f)' % (human_string, score))
	# 		if boo:
	# 			prediction = human_string
	# 			prediction_score = score
	# 			boo = False
	#
	# return prediction, prediction_score

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


def sliding_window(image, stepSize, windowSize):
	# slide window across the image
    for y in xrange(0, image.shape[0], stepSize):
		for x in xrange(0, image.shape[1], stepSize):
			yield(x, y, image[y:y+windowSize[1], x:x+windowSize[0]])

def create_heatmap(image, stepSize, windowSize):

	heatmap = np.zeros((img.shape[0], img.shape[1], 6))
	countmap = np.zeros((img.shape[0], img.shape[1]))

	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			window = image[y:y+windowSize[1], x:x+windowSize[0]]
			if window.shape[0] != windowSize[0] or \
							window.shape[1] != windowSize[1]:
				continue

			cv2.imwrite('window.jpeg', window)
			predictions = label_image('window.jpeg')

			# print check
			# # show steps
			# print x,y
			# print print_prediction(predictions)
			# # windowimg = cv2.imread('window.jpeg')
			# # check = cv2.imshow('Window', windowimg)
			# # print check
			# # raw_input("Press Enter to continue....")
			# # cv2.destroyAllWindows()
			# raw_input("Press Enter to continue")

			os.remove('window.jpeg')

			for n in range(windowSize[1]):
				for m in range(windowSize[0]):
					heatmap[y+n][x+m] += predictions[0]
					countmap[y+n][x+m] += 1

			# sys.exit()

	return heatmap, countmap

# while result == "":

	# colour_img, img_with_matches, img_with_homography, points = chessboard_homography()
window_x = 100
window_y = 100
stepSize = 20
imgpath = 'kinect_images/top_down/start/front_black/camera_image1.jpeg'
img = cv2.imread(imgpath)

with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as f:
	graph_def = tf.GraphDef()
	graph_def.ParseFromString(f.read())
	_ = tf.import_graph_def(graph_def, name='graph1')

heatmap, countmap = create_heatmap(img, stepSize, (window_x, window_y))

# heatmap = heatmap / np.linalg.norm(heatmap)

kingmap = np.zeros((img.shape[0], img.shape[1]))

for x in range(kingmap.shape[0]):
	for y in range(kingmap.shape[1]):
		kingmap[x][y] = heatmap[x][y][2]

print "at heatmap creation"

ax = sns.heatmap(kingmap)
plt.savefig("top_front_black.png")
plt.show()

# print img.shape
# windows = sliding_window(img, step_size, (window_x,window_y))
# count = 0
# for (x,y,window) in windows:
# 	count += 1
# 	print count
# 	print "x: " + str(x)
# 	print "y: " + str(y)
# 	if window.shape[0] != window_x or window.shape[1] != window_y:
# 		continue
# 	cv2.imwrite('sliding_window_test/image'+str(count)+'.jpeg', window)


# 	# Unpersists graph from file for chess piece
# 	with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as f:
# 		graph_def = tf.GraphDef()
# 		graph_def.ParseFromString(f.read())
# 		_ = tf.import_graph_def(graph_def, name='graph1')


# START THE CHESS GAME
# while result == "":
#
# 	colour_img, img_with_matches, img_with_homography, points = chessboard_homography()
#
# 	final_sliding_window(img_with_homography, points, colour_img)
#
# 	filenames = []
# 	square_results = []
# 	# Read all the half sliding windows to detect if there is a piece on the squares or not
# 	for filename in glob.glob('sliding_windows/halves/*.jpg'):
# 		filenames.append(filename)
#
# 	# Sort by natural keys
# 	filenames = sorted(filenames, key=natural_keys)
#
# 	with tf.gfile.FastGFile("retrained_graph_for_square_or_non_square.pb", 'rb') as f:
# 		graph_def = tf.GraphDef()
# 		graph_def.ParseFromString(f.read())
# 		_ = tf.import_graph_def(graph_def, name='graph3')
#
# 	print ".... I'm checking where the pieces are ...."
# 	for filename in filenames:
# 		prediction, score = label_square(filename)
#
# 		if prediction == "square" and score > 0.80:
# 			square_results.append("empty")
#
# 		else:
# 			square_results.append("piece")
#
# 	# print square_results
#
# 	print ".... I'm checking what pieces they are ...."
# 	results = []
# 	filenames = []
# 	# Read all the sliding windows for piece and colour classification
# 	for filename in glob.glob('sliding_windows/*.jpg'):
# 		filenames.append(filename)
#
# 	# Sort by natural keys
# 	filenames = sorted(filenames, key=natural_keys)
#
# 	# Unpersists graph from file for chess piece
# 	with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as f:
# 		graph_def = tf.GraphDef()
# 		graph_def.ParseFromString(f.read())
# 		_ = tf.import_graph_def(graph_def, name='graph1')
#
#
# 	colours = []
# 	counter = 0
# 	print "\n"
# 	for filename in filenames:
# 		# If the square was labeled as empty, ignore it. Otherwise compute piece classification
# 		if square_results[counter] == "piece":
# 			prediction, score = label_image(filename)
#
# 			if score > 0.45:
# 				results.append(prediction)
# 				print prediction, score
#
# 			else:
# 				print "empty", score
# 				results.append("empty")
# 				colours.append("noCol")
#
# 		else:
# 			print "empty"
# 			results.append("empty")
# 			colours.append("noCol")
#
# 		counter += 1
# 		# For better visual feedback, print new_line for every row
# 		if counter%8 == 0 and counter != 0:
# 			print "\n"
#
# 	print ".... I'm checking what colour the pieces I recognised are ...."
# 	print "\n"
# 	# Unpersists graph from file for colour
# 	with tf.gfile.FastGFile("retrained_graph_for_black_and_white.pb", 'rb') as f:
# 		graph_def = tf.GraphDef()
# 		graph_def.ParseFromString(f.read())
# 		_ = tf.import_graph_def(graph_def, name='graph2')
#
# 	colours = []
# 	for c in range(0, 64):
# 		if (results[c] in pieces):
# 			filename_colour = 'sliding_windows/sliding_window' + str(c) + '.jpg'
# 			colour_prediction, colour_score = label_colour(filename_colour)
# 			colours.append(colour_prediction)
# 			print results[c], colour_prediction
# 		else:
# 			colours.append(" ")
# 			print results[c]
# 		c += 1
#
# 	for c in range(0, 64):
# 		if c % 8 == 0:
# 			print "\n"
# 		if results[c] in pieces and colours[c] == "whites":
# 			results[c] = results[c].upper()
# 		if results[c] != "square" or results[c] != "empty":
# 			print results[c], colours[c]
# 		else:
# 			print results[c]
#
# 	#######################
# 	#### Example of results
# 	# results = ["rook","knight","bishop","queen","king","bishop","knight","rook",
# 	# 	   "pawn","pawn","pawn","pawn","pawn","pawn","pawn","pawn",
# 	# 	   "square","square","square","square","square","square","square","square",
# 	# 	   "square","square","square","square","square","square","square","square",
# 	# 	   "square","square","square","square","square","square","square","square",
# 	# 	   "square","square","square","square","square","square","square","square",
# 	# 	   "PAWN","PAWN","PAWN","PAWN","PAWN","PAWN","PAWN","PAWN",
# 	# 	   "ROOK","KNIGHT","BISHOP","QUEEN","KING","BISHOP","KNIGHT","ROOK"]
# 	#######################
#
#
# 	# r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4
# 	current_state_of_the_board = ""
# 	consecutive_empty_square_counter = 0
# 	for c in range(0,64):
# 		if c%8==0:
# 			print ""
# 			# Avoid initial slashbar
# 			if c != 0:
# 				# If there are empty squares on the right edge of the board, save how many
# 				if consecutive_empty_square_counter != 0:
# 					current_state_of_the_board += str(consecutive_empty_square_counter)
# 					consecutive_empty_square_counter = 0
# 				current_state_of_the_board += "/"
# 		if results[c] == "king":
# 			print "k",
# 			if consecutive_empty_square_counter != 0:
# 				current_state_of_the_board += str(consecutive_empty_square_counter)
# 				consecutive_empty_square_counter = 0
# 			current_state_of_the_board += "k"
# 		elif results[c] == "KING":
# 			print "K",
# 			if consecutive_empty_square_counter != 0:
# 				current_state_of_the_board += str(consecutive_empty_square_counter)
# 				consecutive_empty_square_counter = 0
# 			current_state_of_the_board += "K"
# 		elif results[c] == "queen":
# 			print "q",
# 			if consecutive_empty_square_counter != 0:
# 				current_state_of_the_board += str(consecutive_empty_square_counter)
# 				consecutive_empty_square_counter = 0
# 			current_state_of_the_board += "q"
# 		elif results[c] == "QUEEN":
# 			print "Q",
# 			if consecutive_empty_square_counter != 0:
# 				current_state_of_the_board += str(consecutive_empty_square_counter)
# 				consecutive_empty_square_counter = 0
# 			current_state_of_the_board += "Q"
# 		elif results[c] == "knight":
# 			print "n",
# 			if consecutive_empty_square_counter != 0:
# 				current_state_of_the_board += str(consecutive_empty_square_counter)
# 				consecutive_empty_square_counter = 0
# 			current_state_of_the_board += "n"
# 		elif results[c] == "KNIGHT":
# 			print "N",
# 			if consecutive_empty_square_counter != 0:
# 				current_state_of_the_board += str(consecutive_empty_square_counter)
# 				consecutive_empty_square_counter = 0
# 			current_state_of_the_board += "N"
# 		elif results[c] == "bishop":
# 			print "b",
# 			if consecutive_empty_square_counter != 0:
# 				current_state_of_the_board += str(consecutive_empty_square_counter)
# 				consecutive_empty_square_counter = 0
# 			current_state_of_the_board += "b"
# 		elif results[c] == "BISHOP":
# 			print "B",
# 			if consecutive_empty_square_counter != 0:
# 				current_state_of_the_board += str(consecutive_empty_square_counter)
# 				consecutive_empty_square_counter = 0
# 			current_state_of_the_board += "B"
# 		elif results[c] == "pawn":
# 			print "p",
# 			if consecutive_empty_square_counter != 0:
# 				current_state_of_the_board += str(consecutive_empty_square_counter)
# 				consecutive_empty_square_counter = 0
# 			current_state_of_the_board += "p"
# 		elif results[c] == "PAWN":
# 			print "P",
# 			if consecutive_empty_square_counter != 0:
# 				current_state_of_the_board += str(consecutive_empty_square_counter)
# 				consecutive_empty_square_counter = 0
# 			current_state_of_the_board += "P"
# 		elif results[c] == "rook":
# 			print "r",
# 			if consecutive_empty_square_counter != 0:
# 				current_state_of_the_board += str(consecutive_empty_square_counter)
# 				consecutive_empty_square_counter = 0
# 			current_state_of_the_board += "r"
# 		elif results[c] == "ROOK":
# 			print "R",
# 			if consecutive_empty_square_counter != 0:
# 				current_state_of_the_board += str(consecutive_empty_square_counter)
# 				consecutive_empty_square_counter = 0
# 			current_state_of_the_board += "R"
# 		else:
# 			print ".",
# 			consecutive_empty_square_counter += 1
#
# 	if consecutive_empty_square_counter != 0:
# 		current_state_of_the_board += str(consecutive_empty_square_counter)
#
#
# 	chessboard_state_details = " " + returned_state_of_the_board.split(" ", 1)[1]
#
# 	whose_turn = chessboard_state_details[1]
#
# 	current_state_of_the_board += chessboard_state_details
#
# 	if whose_turn == "w":
# 		returned_state_of_the_board, result = my_next_move(current_state_of_the_board)
#
# 	if result != "":
# 		print result
#
#
# 	time.sleep(1)
