
def create_constants():
    labels = []
    with open(labels_path) as image_labels:
        for line in image_labels:
            line = line.strip('\n')
            line = line.replace(" ", "_")
            labels.append(line)
    labels_map = {"black_pawn": "pawn", "black_knight": "knight", "black_bishop": "bishop", "black_king": "king", "black_queen": "queen", "black_rook": "rook",
                "empty_square": "square", "white_pawn": "PAWN", "white_knight": "KNIGHT", "white_bishop": "BISHOP", "white_king": "KING", "white_queen": "QUEEN", "white_rook": "ROOK"}
    piece_count = {"black_pawn": 8, "black_knight": 2, "black_bishop": 2, "black_king": 1, "black_queen": 1, "black_rook": 2,
                "empty_square": 32, "white_pawn": 8, "white_knight": 2, "white_bishop": 2, "white_king": 1, "white_queen": 1, "white_rook": 2}
    letter_count_map = {"p": "black_pawn", "n": "black_knight", "b": "black_bishop", "k": "black_king", "q": "black_queen", "r": "black_rook"
                        ".": "empty_square", "P": "white_pawn", "B": "white_bishop", "N": "white_knight", "K": "white_king", "Q": "white_queen", "R": "white_rook"}
    # For testing
    # piece_count = {"black_pawn": 0, "black_knight": 0, "black_bishop": 0, "black_king": 0, "black_queen": 1, "black_rook": 0,
    # 			"empty_square": 60, "white_pawn": 0, "white_knight": 0, "white_bishop": 0, "white_king": 0, "white_queen": 1, "white_rook": 0}

    board_square_map= {"a1":0 ,"a2":8, "a3":16, "a4":24, "a5":32, "a6":40, "a7":48, "a8":56, 
                        "b1":1 ,"b2":9, "b3":17, "b4":25, "b5":33, "b6":41, "b7":49, "b8":57,
                        "c1":2 ,"c2":10, "c3":18, "c4":26, "c5":34, "c6":42, "c7":50, "c8":58, 
                        "d1":3 ,"d2":11, "d3":19, "d4":27, "d5":35, "d6":43, "d7":51, "d8":59, 
                        "e1":4 ,"e2":12, "e3":20, "e4":28, "e5":36, "e6":44, "e7":52, "e8":60, 
                        "f1":5 ,"f2":13, "f3":21, "f4":29, "f5":37, "f6":45, "f7":53, "f8":61, 
                        "g1":6 ,"g2":14, "g3":22, "g4":30, "g5":38, "g6":46, "g7":54, "g8":62, 
                        "h1":7 ,"h2":15, "h3":23, "h4":31, "h5":39, "h6":47, "h7":55, "h8":63}
    # labels_map = {"pawn": "pawn", "knight": "knight", "bishop": "bishop", "king": "king", "queen": "queen", "rook": "rook", "empty_square": "square"}

    position_map = {}
    right_joint_labels = ['right_s0', 'right_s1',
                    'right_e0', 'right_e1'
                    'right_w0', 'right_w1', 'right_w2']
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

    precedence_list = ["empty_square", "black_king", "black_queen", "white_king", "white_queen", "white_pawn", "black_pawn", 
						"black_knight", "white_bishop", "white_knight", "white_rook", "black_rook", "black_bishop"]
	
	precedence_list = ["empty_square", "black_king", "black_queen", "white_king", "white_queen", "black_pawn", "black_knight", "black_rook", 
						"black_bishop", "white_knight", "white_bishop", "white_pawn", "white_rook"]

    return labels, labels_map, piece_count, board_square_map, position_map, base_right, base_left, precedence_list, letter_count_map