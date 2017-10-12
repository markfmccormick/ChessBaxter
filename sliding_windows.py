#from pyimagesearch.helpers import pyramid
from pyimagesearch.helpers import sliding_window
import time
import cv2
from chess_pieces.pawn_classifier2 import pawn_prediction
from PIL import Image
from label_image import label_image


def my_sliding_window(image,single_square_side,corner1,corner2,corner3,corner4):
	# Define the window width and height
	(winW, winH) = (single_square_side, single_square_side*3/2)
	#pieces = []

	counter = 0
	print "\n"
	for (x, y, window) in sliding_window(image, stepSize=int(single_square_side/1.5), windowSize=(winW, winH)):
		# if the window does not meet our desired window size, ignore it
		if window.shape[0] != winH or window.shape[1] != winW:
			continue

		#im = Image.fromarray(window)
		#im.save("window_"+str(counter)+".jpeg")
		# Draw the window
		cv2.imshow('s',image)
		cv2.waitKey(0)
		cv2.imwrite('sliding_windows/sliding_window'+str(counter)+".jpg",window)
		# label_image(window)
		clone = image.copy()
		cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
		# cv2.imshow("Window", clone)         # window instead of clone to show only the cropped sliding window
		# cv2.waitKey(1)
		counter += 1

	print "\n***** All sliding windows written to file *****\n"

	#return pieces
