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

results = ["rook","knight","bishop","queen","king","bishop","knight","rook",
		   "pawn","pawn","pawn","pawn","pawn","pawn","pawn","pawn",
		   "square","square","square","square","square","square","square","square",
		   "square","square","square","square","square","square","square","square",
		   "square","square","square","square","square","square","square","square",
		   "square","square","square","square","square","square","square","square",
		   "PAWN","PAWN","PAWN","PAWN","PAWN","PAWN","PAWN","PAWN",
		   "ROOK","KNIGHT","BISHOP","QUEEN","KING","BISHOP","KNIGHT","ROOK"]

board_string = create_board_string(results)
# print board_string

board_state_string = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 0"

# moved_board_state_string, game_over = my_next_move(board_state_string) 

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
moved_board_state_string, game_over, best_move  = my_next_move(board_state_string)
