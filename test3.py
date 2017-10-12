from chess_move import my_next_move

# result = my_next_move("7k/8/4b3/7n/8/8/8/7K b KQkq - 0 4")
fen, result = my_next_move("7p/8/8/b4pp1/p2p4/2PP4/1P2P3/2p5 w - - 0 1")

if result != "":
	print "result", result
