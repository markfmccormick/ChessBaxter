import tensorflow as tf, sys
import cv2

# Classify a square as empty or not
def label_square(image_path):

	# Read in the image_data
	image_data3 = tf.gfile.FastGFile(image_path, 'rb').read()
	# image_data = image_path

	# Loads label file, strips off carriage return
	label_lines3 = [line.rstrip() for line
						in tf.gfile.GFile("retrained_labels_for_square_or_non_square.txt")]



	with tf.Session() as sess3:

		# Feed the image_data as input to the graph and get first prediction
		softmax_tensor3 = sess3.graph.get_tensor_by_name('graph3/final_result:0')

		predictions = sess3.run(softmax_tensor3, \
				 {'graph3/DecodeJpeg/contents:0': image_data3})

		# Sort to show labels of first prediction in order of confidence
		top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

		boo = True
		# print "\n"
		for node_id in top_k:
			human_string = label_lines3[node_id]
			score = predictions[0][node_id]
			# print('%s (score = %.5f)' % (human_string, score))
			if boo:
				prediction = human_string
				prediction_score = score
				boo = False

	return prediction, prediction_score
