import tensorflow as tf, sys
import cv2

# Classify an image of a piece as black or white
def label_colour(image_path):

	# Read in the image_data
	image_data2 = tf.gfile.FastGFile(image_path, 'rb').read()
	# image_data = image_path

	# Loads label file, strips off carriage return
	label_lines2 = [line.rstrip() for line
						in tf.gfile.GFile("retrained_labels_for_black_and_white.txt")]



	with tf.Session() as sess2:

		# Feed the image_data as input to the graph and get first prediction
		softmax_tensor2 = sess2.graph.get_tensor_by_name('graph2/final_result:0')

		predictions = sess2.run(softmax_tensor2, \
				 {'graph2/DecodeJpeg/contents:0': image_data2})

		# Sort to show labels of first prediction in order of confidence
		top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

		boo = True
		# print "\n"
		for node_id in top_k:
			human_string = label_lines2[node_id]
			score = predictions[0][node_id]
			# print('%s (score = %.5f)' % (human_string, score))
			if boo:
				prediction = human_string
				prediction_score = score
				boo = False

	return prediction, prediction_score
