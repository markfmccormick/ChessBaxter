import matplotlib as mpl
mpl.use('GTKAgg')

import glob
import re
import time
import os
import sys

import tensorflow as tf
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from final_sliding_window import final_sliding_window
from label_colour import label_colour
from label_image import label_image

from chess_move import my_next_move
from chessboard_detector import chessboard_homography
from label_square import label_square

class Model(object):

	def __init__(self, model_path):
		self.session = tf.Session()

		with tf.gfile.FastGFile(model_path, 'rb') as f:
			graph_def = tf.GraphDef()
			graph_def.ParseFromString(f.read())
			_ = tf.import_graph_def(graph_def, name='graph1')

		self.softmax_tensor = self.session.graph.get_tensor_by_name('graph1/final_result:0')

	def predict(self, image_data):
		# image_data = tf.gfile.FastGFile(image_path, 'rb').read()
		# predictions =  self.session.run(self.softmax_tensor, {'graph1/DecodeJpeg/contents:0': image_data})
		predictions = self.session.run(self.softmax_tensor, {'graph1/DecodeJpeg:0': image_data})
		return predictions

# Natural human sorting
def atoi(text):
	return int(text) if text.isdigit() else text

def natural_keys(text):
	return [ atoi(c) for c in re.split('(\d+)', text) ]

pieces = ["rook","knight","bishop","queen","king","pawn"]

returned_state_of_the_board = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 0"
result = ""

def print_prediction(predictions):
	label_lines = [line.rstrip() for line
						in tf.gfile.GFile("retrained_labels.txt")]
	# Sort to show labels of first prediction in order of confidence
	top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

	boo = True
	# print "\n"
	for node_id in top_k:
		human_string = label_lines[node_id]
		score = predictions[0][node_id]
		# print('%s (score = %.5f)' % (human_string, score))
		if boo:
			prediction = human_string
			prediction_score = score
			boo = False

	return prediction, prediction_score


def sliding_window(image, stepSize, windowSize):
	# slide window across the image
    for y in xrange(0, image.shape[0], stepSize):
		for x in xrange(0, image.shape[1], stepSize):
			yield(x, y, image[y:y+windowSize[1], x:x+windowSize[0]])

def create_heatmap(image, stepSize, windowSize, model_path):

	heatmap = np.zeros((image.shape[0], image.shape[1], 6))
	countmap = np.zeros((image.shape[0], image.shape[1]))

	model = Model(model_path)

	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			window = image[y:y+windowSize[1], x:x+windowSize[0]]
			if window.shape[1] != windowSize[0] or \
							window.shape[0] != windowSize[1]:
				continue

			# cv2.imwrite('window.jpeg', window)
			# predictions = label_image('window.jpeg')
			# predictions = model.predict('window.jpeg')
			window = np.array(window)
			predictions = model.predict(window)

			for n in range(windowSize[1]):
				for m in range(windowSize[0]):
					heatmap[y+n][x+m] += predictions[0]
					countmap[y+n][x+m] += 1

			# sys.exit()

	return heatmap, countmap

def visualise_heatmap(imgpath, model_path):
	window_x = 90
	window_y = 120
	stepSize = 20
	print imgpath
	img = cv2.imread(imgpath)

	heatmap, countmap = create_heatmap(img, stepSize, (window_x, window_y), model_path)

	# heatmap = heatmap / np.linalg.norm(heatmap)

	pawnmap = np.zeros((img.shape[0], img.shape[1]))

	for x in range(pawnmap.shape[0]):
		for y in range(pawnmap.shape[1]):
			pawnmap[x][y] = heatmap[x][y][2]

	print "at heatmap creation"

	ax = sns.heatmap(pawnmap, cbar = False)
	plt.axis('off')
	plt.savefig(imgpath[:-5]+"-map.png", bbox_inches='tight')
	#plt.show()

# while result == "":

	# colour_img, img_with_matches, img_with_homography, points = chessboard_homography()

#imgpath = 'kinect_images/top_down/start/front_black/camera_image1.jpeg'
model_path = "retrained_graph.pb"

imgpaths = glob.glob("kinect_images_new/*_front/*.jpeg")
for path in imgpaths:
	visualise_heatmap(path, model_path)


# print img.shape
# windows = sliding_window(img, step_size, (window_x,window_y))
# count = 0
# for (x,y,window) in windows:
# 	count += 1
# 	print count
# 	print "x: " + str(x)
# 	print "y: " + str(y)
# 	if window.shape[0] != window_x or window.shape[1] != window_y:
# 		continue
# 	cv2.imwrite('sliding_window_test/image'+str(count)+'.jpeg', window)


# 	# Unpersists graph from file for chess piece
# 	with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as f:
# 		graph_def = tf.GraphDef()
# 		graph_def.ParseFromString(f.read())
# 		_ = tf.import_graph_def(graph_def, name='graph1')
