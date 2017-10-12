#! /usr/bin/python
# Copyright (c) 2015, Rethink Robotics, Inc.

# Using this CvBridge Tutorial for converting
# ROS images to OpenCV2 images
# http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython

# Using this OpenCV2 tutorial for saving Images:
# http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_image_display/py_image_display.html

# rospy for the subscriber
import rospy
# ROS Image message
from sensor_msgs.msg import Image
# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError
# OpenCV2 for saving an image
import cv2
import time
from main import play

# Instantiate CvBridge
bridge = CvBridge()
c = 0

def take_picture(msg):

	print("\nReceived an image!")
	try:
		# Convert your ROS Image message to OpenCV2
		cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
	except CvBridgeError, e:
		print(e)
	else:
		# Return image
		cv2.imshow('',cv2_img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		play(cv2_img)

def main2():
	rospy.init_node('image_listener')
	# Define your image topic
	image_topic = "/kinect2/hd/image_color"
	# Set up your publisher - DOES NOT DO ANYTHING FOR NOW AS I NEED TO RUN roslaunch kinect2_bridge kinect2_bridge.launch
	pub = rospy.Publisher(image_topic, Image, queue_size=0)
	# Set up your subscriber and define its callback

	cv2_img = rospy.Subscriber(image_topic, Image, take_picture)


if __name__ == '__main__':
	main()
