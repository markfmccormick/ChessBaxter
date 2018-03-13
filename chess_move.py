import chess
import chess.uci
import stockfish

"""
	Interface to the Stockfish chess engine to work out the best move that can be made with the given board state
"""

# Takes a board state string and uses the chess engine to work out the best move to make
# Returns the best move, a check for the game being over, and the new board state after the move
def my_next_move(state_of_the_chessboard):
	# Initialise the board with its current state
	board = chess.Board(state_of_the_chessboard)
	print "Board before move: "
	print board

	# Check for end of the game
	if board.is_checkmate():
		game_over = "Checkmate, I lost."
	elif board.is_game_over():
		game_over = "Draw"
	else:
		game_over = ""

	fen = str(board.fen).split("\'")[1]

	if game_over != "":
		return fen, game_over


	engine = chess.uci.popen_engine("stockfish-8-linux/Linux/stockfish_8_x64")
	engine.uci()
	engine.ucinewgame()
	engine.position(board)

	try:
		best_move = engine.go(depth=5)[0]
	except RuntimeError:
		raise EngineTerminatedException()
		# return "Invalid state of the chessboard. Not a valid game."
		
	print "best move: ", best_move
	board.push(best_move)
	engine.position(board)
	
	print "Board after move: "
	print board

	# Check for end of the game
	if board.is_checkmate():
		game_over = "Checkmate, I won."
	elif board.is_game_over():
		game_over = "Draw"
	else:
		game_over = ""


	fen = str(board.fen).split("\'")[1]

	return fen, game_over, best_move
