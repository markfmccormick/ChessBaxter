#!/usr/bin/python
# -*- coding: iso-8859-15 -*-

import numpy as np
import cv2
import glob
import os
import fnmatch

from create_board_string import create_board_string
from chess_move import my_next_move
import chess
import chess.uci
import stockfish

# results = ["rook","knight","bishop","queen","king","bishop","knight","rook",
# 		   "pawn","pawn","pawn","pawn","pawn","pawn","pawn","pawn",
# 		   "square","square","square","square","square","square","square","square",
# 		   "square","square","square","square","square","square","square","square",
# 		   "square","square","square","square","square","square","square","square",
# 		   "square","square","square","square","square","square","square","square",
# 		   "PAWN","PAWN","PAWN","PAWN","PAWN","PAWN","PAWN","PAWN",
# 		   "ROOK","KNIGHT","BISHOP","QUEEN","KING","BISHOP","KNIGHT","ROOK"]

# board_string = create_board_string(results)
# # print board_string

# board_state_string = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 0"

# # moved_board_state_string, game_over = my_next_move(board_state_string) 

# square_labels = ["rook","knight","bishop","queen","king","bishop","knight","rook",
# 		"pawn","pawn","pawn","pawn","pawn","pawn","pawn","pawn",
# 		"square","square","square","square","square","square","square","square",
# 		"square","square","square","square","square","square","square","square",
# 		"square","square","square","square","square","square","square","square",
# 		"square","square","square","square","square","square","square","square",
# 		"PAWN","PAWN","PAWN","PAWN","PAWN","PAWN","PAWN","PAWN",
# 		"ROOK","KNIGHT","BISHOP","QUEEN","KING","BISHOP","KNIGHT","ROOK"]

# board_state_string = create_board_string(square_labels)
# board_state_string += " w KQkq - 0 0"
# moved_board_state_string, game_over, best_move  = my_next_move(board_state_string)

position_map = {}
right_joint_labels = ['right_s0', 'right_s1', 'right_e0', 'right_e1’, ’right_w0', 'right_w1', 'right_w2']
with open("square_positions.txt") as position_labels:
	for line in position_labels:
		square_positions = {}
		joint_positions = {}
		line = line.strip('\n')
		line = line.split(":")
		square = line[0]
		positions = line[1].split(";")
		values = positions[1].split(",")
		for i in range(len(right_joint_labels)):
			joint_positions[right_joint_labels[i]] = values[i]
		square_positions[positions[0]] = joint_positions
		values = positions[3].split(",")
		for i in range(len(right_joint_labels)):
			joint_positions[right_joint_labels[i]] = values[i]
		square_positions[positions[2]] = joint_positions
		position_map[square] = square_positions

print position_map