import cv2
import math

#
# Write to HDD 64 images representing each square of the chessboard.
# These images will then be used by tensorflow to predict which piece,
# if any, sits on them.
def final_sliding_window(img, lines_coordinates, colour_img):

	mid_point_top_x = lines_coordinates[26][0]
	mid_point_bottom_x = lines_coordinates[27][0]
	mid_point_average = int(round(mid_point_top_x+mid_point_bottom_x)/2)
	# This circle should be in the top right corner of the board
	# cv2.circle(img, (lines_coordinates[0]), 5, (255, 0, 0), 5)
	# cv2.imshow('',img)
	# print mid_point_average
	global_counter = 0
	points = []
	c = 2
	while c < len(lines_coordinates)/2:
		pt1 = lines_coordinates[c]
		pt2 = lines_coordinates[c+1]
		mean_height = int(round((pt1[1]+pt2[1])/2))
		c += 2

		x2 = int(round(pt1[0]))
		y2 = int(round(pt1[1]))
		x1 = int(round(pt2[0]))
		y1 = int(round(pt2[1]))
		# Eucleadian distance
		first_half_length_of_line = math.sqrt((mid_point_average - x1)**2 + (mean_height - y1)**2)
		second_half_length_of_line = math.sqrt((x2 - mid_point_average)**2 + (y2 - mean_height)**2)
		first_step = int(round(first_half_length_of_line/4))
		second_step = int(round(second_half_length_of_line/4))
		current_point = x1
		c2 = 0
		points = []
		while c2 < 5:
			points.append([current_point,mean_height])
			c2 += 1
			current_point = current_point + first_step

		while c2 < 9:
			points.append([current_point,mean_height])
			c2 += 1
			current_point = current_point + second_step


		c3 = 0
		while c3 < 8:
			# change the height of the sliding window according to the distance from the camera
			# (the higher c, the closer to the camera, the bigger the sliding_window)
			if c < 18:

				sliding_window = img[mean_height-110:mean_height, points[c3][0]:points[c3+1][0]]
				colour_sliding_window = colour_img[mean_height-110:mean_height, points[c3][0]:points[c3+1][0]]
				half_sliding_window = colour_img[mean_height-55:mean_height, points[c3][0]:points[c3+1][0]]

			else:
				sliding_window = img[mean_height-130:mean_height, points[c3][0]:points[c3+1][0]]
				colour_sliding_window = colour_img[mean_height-130:mean_height, points[c3][0]:points[c3+1][0]]
				half_sliding_window = colour_img[mean_height-70:mean_height, points[c3][0]:points[c3+1][0]]
			print c
			# cv2.imshow("sliding_window",sliding_window)
			# cv2.waitKey(0)
			cv2.imwrite('sliding_windows/sliding_window'+str(global_counter)+".jpg",colour_sliding_window)
			cv2.imwrite('sliding_windows/halves/sliding_window'+str(global_counter)+".jpg",half_sliding_window)
			c3 += 1
			global_counter += 1
