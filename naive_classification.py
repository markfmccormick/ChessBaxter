import cv2
import numpy as np
import chess
from create_board_string import create_board_string

"""
	Separated from the main program and put together in one place for readability.
	Used to evaluate the effectiveness of all the naive classification systems tested
	but not used in the main program.
"""

def show_naive_classification(center_keypoints, heatmap, countmap, labels, labels_map):
    square_labels = label_squares_point(center_keypoints, heatmap, countmap, labels, labels_map)
    board_state_string = create_board_string(square_labels)
    board_state_string += " w KQkq - 0 0"
    board = chess.Board(board_state_string)
    print "Method - Center point: "
    print board

    square_labels = label_squares_box_total(center_keypoints, heatmap, countmap, labels, labels_map)
    board_state_string = create_board_string(square_labels)
    board_state_string += " w KQkq - 0 0"
    board = chess.Board(board_state_string)
    print "Method - Box total: "
    print board

    square_labels = label_squares_peak(center_keypoints, heatmap, countmap, labels, labels_map)
    board_state_string = create_board_string(square_labels)
    board_state_string += " w KQkq - 0 0"
    board = chess.Board(board_state_string)
    print "Method - Peak value: "
    print board

    square_labels = label_squares_center_weighted(center_keypoints, heatmap, countmap, labels, labels_map)
    board_state_string = create_board_string(square_labels)
    board_state_string += " w KQkq - 0 0"
    board = chess.Board(board_state_string)
    print "Method - Center weighted: "
    print board

    square_labels = label_squares_box_total_threshold(center_keypoints, heatmap, countmap, labels, labels_map)
    board_state_string = create_board_string(square_labels)
    board_state_string += " w KQkq - 0 0"
    board = chess.Board(board_state_string)
    print "Method - Box total threshold: "
    print board

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