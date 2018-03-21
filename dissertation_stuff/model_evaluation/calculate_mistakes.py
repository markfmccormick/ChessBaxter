import glob, os, sys, fnmatch

base_path = "data/"

checks = []
for root, dirnames, filenames in os.walk(base_path):
    for filename in fnmatch.filter(filenames, 'Perfect.txt'):
        checks.append(os.path.join(root, filename))

results = []
for root, dirnames, filenames in os.walk(base_path):
    for filename in fnmatch.filter(filenames, 'output.txt'):
        results.append(os.path.join(root, filename))

def read_board(resultf):
    i = 0
    board = []
    while i < 8:
        line = resultf.readline().strip("\n")
        for l in line:
            if l != " ":
                board.append(l)
        i += 1
    return board


for result in results:
    check = [i for i in checks if i.split("/")[1] == result.split("/")[1]][0]
    path = result[:-10]
    checkf = open(check, 'r')
    resultf = open(result, 'r')
    outputf = open(path + "result.txt", 'w')
    board = []
    for line in checkf:
        line = line.strip("\n")
        for l in line:
            if l != " ":
                board.append(l)

    board_label = []
    resultf.readline()
    outputf.write("Results: ")
    for i in range(5):
        wrong = 0
        name = resultf.readline()
        outputf.write(name)
        board_label = read_board(resultf)
        for i, sq in enumerate(board):
            if sq != board_label[i]:
                wrong += 1
        outputf.write("Mistakes: "+str(wrong)+"\n")

    checkf.close()
    resultf.close()
    outputf.close()