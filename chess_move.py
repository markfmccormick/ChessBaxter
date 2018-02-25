import chess
import chess.uci
import stockfish

def my_next_move(state_of_the_chessboard):
	# To HACK, uncomment one of the states of the chessboard
	# state_of_the_chessboard = "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4"		# Initial state
	# state_of_the_chessboard = "7k/8/4p3/b4pp1/p7/2P5/1P2P3/7K b - - 40 40"
	# state_of_the_chessboard = "8/8/4b3/7n/8/8/8/8 b KQkq - 0 4"

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
		best_move = engine.go(depth=10)[0]
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
