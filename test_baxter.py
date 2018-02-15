import chess
import chess.uci
import stockfish

from create_board_string import create_board_string
from chess_move import my_next_move
from move_baxter import perform_move

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

position_map = {}
right_joint_labels = ['right_s0', 'right_s1', 'right_e0', 'right_e1', 'right_w0', 'right_w1', 'right_w2']
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
"""
while True:
	user_move = raw_input("Enter the move you made: ")
	initial_square = user_move[0:2]
	final_square = user_move[2:4]
	perform_move(initial_square, final_square, position_map)
"""

while game_over == "":
	moved_board_state_string, game_over, best_move = my_next_move(board_state_string)
	initial_square = best_move.uci()[0:2]
	final_square = best_move.uci()[2:4]
	print initial_square, final_square
	print "Move made: "+best_move.uci()

	# Baxter makes the move
	perform_move(initial_square, final_square, position_map)

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
