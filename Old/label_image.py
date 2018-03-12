import tensorflow as tf, sys
import cv2

# Classify a chess piece
def label_image(image_path):

	# Read in the image_data
	image_data = tf.gfile.FastGFile(image_path, 'rb').read()
	# image_data = image_path

	# Loads label file, strips off carriage return
	label_lines = [line.rstrip() for line
						in tf.gfile.GFile("retrained_labels.txt")]



	with tf.Session() as sess:

		# Feed the image_data as input to the graph and get first prediction
		softmax_tensor = sess.graph.get_tensor_by_name('graph1/final_result:0')

		predictions = sess.run(softmax_tensor, \
				 {'graph1/DecodeJpeg/contents:0': image_data})

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
