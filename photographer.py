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
import glob
import re

# Instantiate CvBridge
bridge = CvBridge()
new_image_counter = 0

# Natural human sorting
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]

# Must launch 'roslaunch kinect2_bridge kinect2_bridge.launch'

def image_callback(msg):

    path = "board_images"
    print "\nReceived an image!"
    try:
        # Convert ROS image message to openCV2
        cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
    except CvBridgeError, e:
        print e
    else:
        # Save image in jpeg format, limit folder to max 100 pictures length
        global counter
        global new_image_counter
        if new_image_counter == 100:
            new_image_counter = 0
        new_image_counter += 1
        print new_image_counter
        cv2.imwrite(path+'/camera_image' + str(new_image_counter) + '.jpeg', cv2_img)
        time.sleep(2)


def main():
    global new_image_counter
    filenames = []
    # Read all filenames in kinect_images
    for filename in glob.glob('kinect_images/*.jpeg'):
        filenames.append(filename)
    # Sort with natural sorting
    filenames = sorted(filenames, key=natural_keys)

    if len(filenames) > 0:
        new_image_counter = int(''.join([i for i in filenames[-1] if i.isdigit()])) + 1

    rospy.init_node('image_listener')
    # Define image topic
    image_topic = "/kinect2/hd/image_color"
    # Set up publisher - Need to run roslaunch kinect2_bridge kinect2_bridge.launch in a terminal
    pub = rospy.Publisher(image_topic, Image, queue_size = 0)
    # Set up subscriber and define it's callback function
    rospy.Subscriber(image_topic, Image, image_callback)
    # Run until ctrl + c
    rospy.spin()

if __name__ == '__main__':
    main()









