import math
import cv2
import numpy as np
from final_sliding_window import final_sliding_window
from chessboard_detector import chessboard_homography


# Get the picture of a chessboard with the outer corners' coordinates
colour_img, img_with_matches, img, lines_coordinates = chessboard_homography()

# final_sliding_window(img_with_homography, points, colour_img)

mid_point_top_x = lines_coordinates[26][0]
mid_point_bottom_x = lines_coordinates[27][0]
mid_point_average = int(round(mid_point_top_x+mid_point_bottom_x)/2)

global_counter = 0
points = []
c = 2
while c < len(lines_coordinates)/2:

	pt1 = lines_coordinates[c]
	pt2 = lines_coordinates[c+1]

	mean_height = int(round((pt1[1]+pt2[1])/2))
	c += 2

	pt3 = int(round(pt1[0]))
	pt4 = int(round(pt1[1]))
	pt5 = int(round(pt2[0]))
	pt6 = int(round(pt2[1]))

	first_half_length_of_line = math.sqrt((mid_point_average - pt5)**2 + (mean_height - pt6)**2)

	second_half_length_of_line = math.sqrt((pt3 - mid_point_average)**2 + (pt4 - mean_height)**2)
	first_step = int(round(first_half_length_of_line/4))
	second_step = int(round(second_half_length_of_line/4))

	current_point = pt5
	c2 = 0

	points = []
	while c2 < 5:
		# cv2.circle(img, ((current_point),(mean_height)), 5, (0, 0, 0), 5)
		points.append([current_point,mean_height])
		c2 += 1
		current_point = current_point + first_step

	while c2 < 9:
		# cv2.circle(img, ((current_point),(mean_height)), 5, (0, 0, 0), 5)
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
			sliding_window = img[mean_height-120:mean_height, points[c3][0]:points[c3+1][0]]
			colour_sliding_window = colour_img[mean_height-120:mean_height, points[c3][0]:points[c3+1][0]]
			half_sliding_window = colour_img[mean_height-70:mean_height, points[c3][0]:points[c3+1][0]]

		cv2.imshow("sliding_window",half_sliding_window)
		cv2.waitKey(1000)
		cv2.destroyAllWindows()
		# cv2.imshow("sliding_window",colour_sliding_window)
		# cv2.waitKey(1500)
		# cv2.destroyAllWindows()

		empty_square = raw_input("Is this an empty square? ")
		if empty_square == "y":
			cv2.imwrite('/home/lorenzo/tf_files/square_or_no_square/square/added/b'+str(global_counter)+".jpg",half_sliding_window)
		elif empty_square == "n":
			cv2.imwrite('/home/lorenzo/tf_files/square_or_no_square/no_square/added/b'+str(global_counter)+".jpg",half_sliding_window)
			cv2.imshow("sliding_window",colour_sliding_window)
			cv2.waitKey(750)
			cv2.destroyAllWindows()
			class_name = raw_input("What class is this picture? ")
			cv2.imwrite('/home/lorenzo/tf_files/chess_pieces/'+class_name+'/added/'+'b'+str(global_counter)+".jpg",colour_sliding_window)
			colour = raw_input("What colour is it? ")
			if colour == "w":
				cv2.imwrite('/home/lorenzo/tf_files/black_or_white/whites/added/'+'b'+str(global_counter)+".jpg",colour_sliding_window)
			elif colour == "b":
				cv2.imwrite('/home/lorenzo/tf_files/black_or_white/blacks/added/'+'b'+str(global_counter)+".jpg",colour_sliding_window)

		c3 += 1
		global_counter += 1
