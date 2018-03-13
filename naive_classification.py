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

"""
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
    print "Method - Window around center: "
    print board


    square_labels = square_classification_smart_zeroing(center_keypoints, heatmap, countmap, labels, labels_map, piece_count)
    board_state_string = create_board_string(square_labels)
    board_state_string += " w KQkq - 0 0"
    board = chess.Board(board_state_string)
    print "Method - Box total smart zeroing: "
    print board
    """

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


"""
def square_classification_smart_zeroing(center_keypoints, heatmap, countmap, labels, labels_map, piece_count_master):
	piece_count = dict(piece_count_master)
	square_labels = ["" for i in range(64)]
	precedence_list = ["empty_square", "black_king", "black_queen", "white_king", "white_queen", "black_rook", "white_rook",
						"white_pawn", "black_pawn", "white_bishop", "white_knight", "black_knight", "black_bishop"]
	
	for piece in precedence_list:
		piece_index = labels.index(piece)
		# square_data = [0 if square_labels[j] == "" else -1 for j in range(len(square_labels))]
		square_data = []
		for point in center_keypoints:
			total = 0
			for y in range(int(point[0])-5, int(point[0])+5):
				for x in range(int(point[1])-5, int(point[1])+5):
					total += heatmap[x][y][piece_index]
			square_data.append(total)
		for count in range(piece_count[piece]):
			max_index = -1
			max_value = -1
			for i in range(len(square_data)):
				if square_data[i] > max_value and square_labels[i] == "":
					max_index = i
					max_value = square_data[i]
			square_labels[max_index] = labels_map[labels[piece_index]]
			square_data[max_index] = 0
		for y in range(len(heatmap)):
			for x in range(len(heatmap[y])):
				heatmap[y][x][piece_index] = 0
"""