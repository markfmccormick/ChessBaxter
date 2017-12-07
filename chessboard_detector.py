import numpy as np
import cv2
# from matplotlib import pyplot as plt
import glob
import time
import rospy
import re


image_counter = 0

def atoi(text):
	return int(text) if text.isdigit() else text

def natural_keys(text):
	return [ atoi(c) for c in re.split('(\d+)', text) ]


def drawMatches(img1, kp1, img2, kp2, matches):
	"""
	My own implementation of cv2.drawMatches as OpenCV 2.4.9
	does not have this function available but it's supported in
	OpenCV 3.0.0

	This function takes in two images with their associated
	keypoints, as well as a list of DMatch data structure (matches)
	that contains which keypoints matched in which images.

	An image will be produced where a montage is shown with
	the first image followed by the second image beside it.

	Keypoints are delineated with circles, while lines are connected
	between matching keypoints.

	img1,img2 - Grayscale images
	kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint
			  detection algorithms
	matches - A list of matches of corresponding keypoints through any
			  OpenCV keypoint matching algorithm
	"""

	# Create a new output image that concatenates the two images together
	# (a.k.a) a montage
	rows1 = img1.shape[0]
	cols1 = img1.shape[1]
	rows2 = img2.shape[0]
	cols2 = img2.shape[1]

	out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

	# Place the first image to the left
	out[:rows1,:cols1,:] = np.dstack([img1, img1, img1])

	# Place the next image to the right of it
	out[:rows2,cols1:cols1+cols2,:] = np.dstack([img2, img2, img2])

	# For each pair of points we have between both images
	# draw circles, then connect a line between them
	for mat in matches:

		# Get the matching keypoints for each of the images
		img1_idx = mat.queryIdx
		img2_idx = mat.trainIdx

		# x - columns
		# y - rows
		(x1,y1) = kp1[img1_idx].pt
		(x2,y2) = kp2[img2_idx].pt

		# Draw a small circle at both co-ordinates
		# radius 4
		# colour blue
		# thickness = 1
		cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)
		cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

		# Draw a line in between the two points
		# thickness = 1
		# colour blue
		cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)

	# cv2.imshow('Matched Features', out)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	# Show the image
	return out



def chessboard_segmentation(img1):
	points = []
	list_of_points = []
	with open('coordinates.txt') as coordinates:
		for line in coordinates:
			line = line.split()
			for number in line:
				points = points + [number]
	c = 0

	while(c<len(points)):
		pt1 = (int(points[c]),int(points[c+1]))
		pt2 = (int(points[c+2]),int(points[c+3]))
		# cv2.line(img1,pt1,pt2,(0,0,0), 2)
		list_of_points.append(pt1)
		list_of_points.append(pt2)
		c += 4

	#cv2.imshow('gray', img1)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
	return list_of_points

def chessboard_homography():
	MIN_MATCH_COUNT = 10

	img1 = cv2.imread('my_chessboard.png',0)

	# Retrieve list of files in the folder where the pictures are
	list_of_pictures = sorted(glob.glob("kinect_images/*.jpeg"),key=natural_keys)
	list_of_counters = []
	for picture in list_of_pictures:
		counter = picture.split('camera_image')[1]
		counter = counter.split('.')[0]
		list_of_counters.append(counter)

	print list_of_pictures
	print list_of_counters

	# If there exist files - ie pictures - in that folder, retrieve the latest one
	if(len(list_of_pictures)!=0):
		# Take the last picture taken and its number to process it
		counter = list_of_counters[-1]
		img2 = cv2.imread('kinect_images/camera_image'+str(counter)+'.jpeg',0)
		colour_img = cv2.imread('kinect_images/camera_image'+str(counter)+'.jpeg')
		# print counter
		# cv2.imshow("i",img2)
		# cv2.waitKey(0)


	# Else loop until a picture appears in that folder
	else:
		while(len(list_of_pictures)==0):
			print ("...waiting for a picture...")
			time.sleep(5)
			list_of_pictures = sorted(glob.glob("kinect_images/*.jpeg"),key=natural_keys)
			list_of_counters = []
			for picture in list_of_pictures:
				counter = picture.split('camera_image')[1]
				counter = counter.split('.')[0]
				list_of_counters.append(counter)

			if(len(list_of_pictures)!=0):
				# Take the last picture taken and its number to process it
				counter = list_of_counters[-1]
				img2 = cv2.imread('kinect_images/camera_image'+str(counter)+'.jpeg',0)


	#rospy.init_node('image_processor')


	# Initiate SIFT detector
	#siftkp = cv2.FeatureDetector_create("SIFT")
	#siftdesc = cv2.DescriptorExtractor_create("SIFT")
	siftdesc = cv2.xfeatures2d.SIFT_create()

	(kp1, des1) = siftdesc.detectAndCompute(img1, None)
	#kp1 = siftkp.detect(img1)
	#(kp1, des1) = siftdesc.compute(img1, kp1)

	(kp2, des2) = siftdesc.detectAndCompute(img2, None)
	#kp2 = siftkp.detect(img2)
	#(kp2, des2) = siftdesc.compute(img2, kp2)


	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks = 50)

	flann = cv2.FlannBasedMatcher(index_params, search_params)

	matches = flann.knnMatch(des1,des2,k=2)

	# store all the good matches as per Lowe's ratio test.
	good = []
	for m,n in matches:
		if m.distance < 0.7*n.distance:
			good.append(m)



	if len(good)>MIN_MATCH_COUNT:
		src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
		dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

		M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
		matchesMask = mask.ravel().tolist()

		h,w = img1.shape
		pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
		dst = cv2.perspectiveTransform(pts,M)
		# cv2.polylines(img2,[np.int32(dst)],True,255,3)
		list_of_points = chessboard_segmentation(img1)
		pts2 = np.float32(list_of_points).reshape(-1,1,2)
		dst2 = cv2.perspectiveTransform(pts2,M)

		# Insert each point in a list of tuples: [(x1,y1),(x2,y2)...(xn,yn)]
		actual_list_of_points = []
		for point in dst2:
			actual_list_of_points.append((point[0][0],point[0][1]))

		#######################
		######## HACK  ########
		#######################
		actual_list_of_points = [(1133, 533), (661, 523), (1141, 569), (648,562), (1154,610), (635,603),
								 (1164,654), (619,647), (1178,705), (602,698), (1193,761), (582,756),
								 (1209,827), (561,819), (1227,897), (535,894), (1250,981), (508,978),
								 (1133,533), (1250,981), (1074,532), (1157,980), (1016,531), (1068,980),
								 (959,530), (975,979), (898,528), (886,979), (841,527), (791,978),
								 (781,526), (699,979), (723,524), (603,977), (661,523), (508,978)]
		# Draw a line between every pair of points
		c = 0

		while(c<len(actual_list_of_points)-1):
			pt1 = actual_list_of_points[c]
			pt2 = actual_list_of_points[c+1]
			# cv2.line(img2,pt1,pt2,0,3)
			c += 2


	else:
		print ("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
		matchesMask = None


	draw_params = dict(matchColor = (0,255,0), # draw matches in green color
					   singlePointColor = None,
					   matchesMask = matchesMask, # draw only inliers
					   flags = 2)


	img3 = drawMatches(img1,kp1,img2,kp2,good)

	# print actual_list_of_points
	return colour_img, img3, img2, actual_list_of_points
