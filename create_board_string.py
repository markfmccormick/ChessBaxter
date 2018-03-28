"""
    Takes the 64 length array of board position piece classifications and
    constructs the FEN notation for the board state
    see: https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation

	Example of results, the input to the function
	results = ["rook","knight","bishop","queen","king","bishop","knight","rook",
		   "pawn","pawn","pawn","pawn","pawn","pawn","pawn","pawn",
		   "square","square","square","square","square","square","square","square",
		   "square","square","square","square","square","square","square","square",
		   "square","square","square","square","square","square","square","square",
		   "square","square","square","square","square","square","square","square",
		   "PAWN","PAWN","PAWN","PAWN","PAWN","PAWN","PAWN","PAWN",
		   "ROOK","KNIGHT","BISHOP","QUEEN","KING","BISHOP","KNIGHT","ROOK"]
"""

def create_board_string(results):
    current_state_of_the_board = ""
    consecutive_empty_square_counter = 0

    for c in range(0,64):
        if c%8 == 0:
            if c != 0:
                if consecutive_empty_square_counter != 0:
                    current_state_of_the_board += str(consecutive_empty_square_counter)
                    consecutive_empty_square_counter = 0
                current_state_of_the_board += "/"
        if results[c] == "king":
            if consecutive_empty_square_counter != 0:
                current_state_of_the_board += str(consecutive_empty_square_counter)
                consecutive_empty_square_counter = 0
            current_state_of_the_board += "k"
        elif results[c] == "KING":
            if consecutive_empty_square_counter != 0:
                current_state_of_the_board += str(consecutive_empty_square_counter)
                consecutive_empty_square_counter = 0
            current_state_of_the_board += "K"
        elif results[c] == "queen":
            if consecutive_empty_square_counter != 0:
                current_state_of_the_board += str(consecutive_empty_square_counter)
                consecutive_empty_square_counter = 0
            current_state_of_the_board += "q"
        elif results[c] == "QUEEN":
            if consecutive_empty_square_counter != 0:
                current_state_of_the_board += str(consecutive_empty_square_counter)
                consecutive_empty_square_counter = 0
            current_state_of_the_board += "Q"
        elif results[c] == "knight":
            if consecutive_empty_square_counter != 0:
                current_state_of_the_board += str(consecutive_empty_square_counter)
                consecutive_empty_square_counter = 0
            current_state_of_the_board += "n"
        elif results[c] == "KNIGHT":
            if consecutive_empty_square_counter != 0:
                current_state_of_the_board += str(consecutive_empty_square_counter)
                consecutive_empty_square_counter = 0
            current_state_of_the_board += "N"
        elif results[c] == "bishop":
            if consecutive_empty_square_counter != 0:
                current_state_of_the_board += str(consecutive_empty_square_counter)
                consecutive_empty_square_counter = 0
            current_state_of_the_board += "b"
        elif results[c] == "BISHOP":
            if consecutive_empty_square_counter != 0:
                current_state_of_the_board += str(consecutive_empty_square_counter)
                consecutive_empty_square_counter = 0
            current_state_of_the_board += "B"
        elif results[c] == "pawn":
            if consecutive_empty_square_counter != 0:
                current_state_of_the_board += str(consecutive_empty_square_counter)
                consecutive_empty_square_counter = 0
            current_state_of_the_board += "p"
        elif results[c] == "PAWN":
            if consecutive_empty_square_counter != 0:
                current_state_of_the_board += str(consecutive_empty_square_counter)
                consecutive_empty_square_counter = 0
            current_state_of_the_board += "P"
        elif results[c] == "rook":
            if consecutive_empty_square_counter != 0:
                current_state_of_the_board += str(consecutive_empty_square_counter)
                consecutive_empty_square_counter = 0
            current_state_of_the_board += "r"
        elif results[c] == "ROOK":
            if consecutive_empty_square_counter != 0:
                current_state_of_the_board += str(consecutive_empty_square_counter)
                consecutive_empty_square_counter = 0
            current_state_of_the_board += "R"
        else:
            consecutive_empty_square_counter += 1
    if consecutive_empty_square_counter != 0:
        current_state_of_the_board += str(consecutive_empty_square_counter)
    
    return current_state_of_the_board
