import chess
import chess.uci
import stockfish
import rospy
import baxter_interface
import time

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

board_square_map= {"a1":0 ,"a2":1, "a3":2, "a4":3, "a5":4, "a6":5, "a7":6, "a8":7, 
					"b1":8 ,"b2":9, "b3":10, "b4":11, "b5":12, "b6":13, "b7":14, "b8":15,
					"c1":16 ,"c2":17, "c3":18, "c4":19, "c5":20, "c6":21, "c7":22, "c8":23, 
					"d1":24 ,"d2":25, "d3":26, "d4":27, "d5":28, "d6":29, "d7":30, "d8":31, 
					"e1":32 ,"e2":33, "e3":34, "e4":35, "e5":36, "e6":37, "e7":38, "e8":39, 
					"f1":40 ,"f2":41, "f3":42, "f4":43, "f5":44, "f6":45, "f7":46, "f8":47, 
					"g1":48 ,"g2":49, "g3":50, "g4":51, "g5":52, "g6":53, "g7":54, "g8":55, 
					"h1":56 ,"h2":57, "h3":58, "h4":59, "h5":60, "h6":61, "h7":62, "h8":63}

pivot_points = ["a8", "a7", ]

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

position_map["pivot_from"] = {"above": {'right_s0': 0.7, 'right_s1': -0.3,
	    								'right_e0': 0.0, 'right_e1': 0.7,
	    								'right_w0': 0.0, 'right_w1': 0.5, 'right_w2': 0.0}, 
								"on": {'right_s0': 0.7, 'right_s1': -0.29,
	    							'right_e0': 0.0, 'right_e1': 1.2,
	    							'right_w0': 0.0, 'right_w1': 0.0, 'right_w2': 0.0}}
position_map["pivot_to"] = {"above": {'right_s0': 0.7, 'right_s1': -0.13,
	    								'right_e0': 0.0, 'right_e1': 0.0,
	    								'right_w0': 0.0, 'right_w1': 1.5, 'right_w2': 0.0}, 
								"on": {'right_s0': 0.7, 'right_s1': 0.165,
	    								'right_e0': 0.0, 'right_e1': 0.0,
										'right_w0': 0.0, 'right_w1': 1.2, 'right_w2': 0.0}}

position_map["capture"] = {"above": {'right_s0': 0.85, 'right_s1': -0.13,
	    								'right_e0': 0.0, 'right_e1': 0.0,
	    								'right_w0': 0.0, 'right_w1': 1.5, 'right_w2': 0.0}, 
								"on": {'right_s0': 0.85, 'right_s1': 0.165,
	    								'right_e0': 0.0, 'right_e1': 0.0,
	    								'right_w0': 0.0, 'right_w1': 1.2, 'right_w2': 0.0}}

base_right = {'right_s0': 0.08, 'right_s1': -1.0, 
	'right_e0': 1.19, 'right_e1': 1.94, 
	'right_w0': -0.67, 'right_w1': 1.03, 'right_w2': 0.50}
base_left = {'left_s0': 1.0, 'left_s1': -1.0, 
	'left_e0': -1.19, 'left_e1': 1.94, 
	'left_w0': 0.67, 'left_w1': 1.03, 'left_w2': -0.50}

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

while game_over == "":
	moved_board_state_string, game_over, best_move = my_next_move(board_state_string)
	board = chess.Board(moved_board_state_string)
	castling = board.is_castling(best_move)

	initial = best_move.uci()[0:2]
	final = best_move.uci()[2:4]
	print initial_square, final_square
	print "Move made: "+best_move.uci()
	print "Board after move: "
	print board

	# Perform move with Baxter
	if castling == False:
    	pivot = ""
    	capture = False
		if board.piece_at(board_square_map[final]) != "":
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

	if board.is_checkmate():
			game_over = "Checkmate, I lost."
	elif board.is_game_over():
		game_over = "Draw"
	else:
		game_over = ""
	
	board_state_string = moved_board_state_string

	"""
	// For user inputed moves
	user_move = raw_input("Enter the move you made: ")
	move = chess.Move.from_uci(user_move)
	board.push(move)
	board_state_string = str(board.fen).split("\'")[1]
	print "Board after user move: "
	print board
	"""
