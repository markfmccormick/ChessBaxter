# Using Lorenzo's code for this, so conforming to the syntax he used
def create_board_string(results):
    current_state_of_the_board = ""
	consecutive_empty_square_counter = 0
	for c in range(0,64):
		if c%8==0:
			print ""
			# Avoid initial slashbar
			if c != 0:
				# If there are empty squares on the right edge of the board, save how many
				if consecutive_empty_square_counter != 0:
					current_state_of_the_board += str(consecutive_empty_square_counter)
					consecutive_empty_square_counter = 0
				current_state_of_the_board += "/"
		if results[c] == "king":
			print "k",
			if consecutive_empty_square_counter != 0:
				current_state_of_the_board += str(consecutive_empty_square_counter)
				consecutive_empty_square_counter = 0
			current_state_of_the_board += "k"
		elif results[c] == "KING":
			print "K",
			if consecutive_empty_square_counter != 0:
				current_state_of_the_board += str(consecutive_empty_square_counter)
				consecutive_empty_square_counter = 0
			current_state_of_the_board += "K"
		elif results[c] == "queen":
			print "q",
			if consecutive_empty_square_counter != 0:
				current_state_of_the_board += str(consecutive_empty_square_counter)
				consecutive_empty_square_counter = 0
			current_state_of_the_board += "q"
		elif results[c] == "QUEEN":
			print "Q",
			if consecutive_empty_square_counter != 0:
				current_state_of_the_board += str(consecutive_empty_square_counter)
				consecutive_empty_square_counter = 0
			current_state_of_the_board += "Q"
		elif results[c] == "knight":
			print "n",
			if consecutive_empty_square_counter != 0:
				current_state_of_the_board += str(consecutive_empty_square_counter)
				consecutive_empty_square_counter = 0
			current_state_of_the_board += "n"
		elif results[c] == "KNIGHT":
			print "N",
			if consecutive_empty_square_counter != 0:
				current_state_of_the_board += str(consecutive_empty_square_counter)
				consecutive_empty_square_counter = 0
			current_state_of_the_board += "N"
		elif results[c] == "bishop":
			print "b",
			if consecutive_empty_square_counter != 0:
				current_state_of_the_board += str(consecutive_empty_square_counter)
				consecutive_empty_square_counter = 0
			current_state_of_the_board += "b"
		elif results[c] == "BISHOP":
			print "B",
			if consecutive_empty_square_counter != 0:
				current_state_of_the_board += str(consecutive_empty_square_counter)
				consecutive_empty_square_counter = 0
			current_state_of_the_board += "B"
		elif results[c] == "pawn":
			print "p",
			if consecutive_empty_square_counter != 0:
				current_state_of_the_board += str(consecutive_empty_square_counter)
				consecutive_empty_square_counter = 0
			current_state_of_the_board += "p"
		elif results[c] == "PAWN":
			print "P",
			if consecutive_empty_square_counter != 0:
				current_state_of_the_board += str(consecutive_empty_square_counter)
				consecutive_empty_square_counter = 0
			current_state_of_the_board += "P"
		elif results[c] == "rook":
			print "r",
			if consecutive_empty_square_counter != 0:
				current_state_of_the_board += str(consecutive_empty_square_counter)
				consecutive_empty_square_counter = 0
			current_state_of_the_board += "r"
		elif results[c] == "ROOK":
			print "R",
			if consecutive_empty_square_counter != 0:
				current_state_of_the_board += str(consecutive_empty_square_counter)
				consecutive_empty_square_counter = 0
			current_state_of_the_board += "R"
		else:
			print ".",
			consecutive_empty_square_counter += 1

	if consecutive_empty_square_counter != 0:
		current_state_of_the_board += str(consecutive_empty_square_counter)

    return current_state_of_the_board
