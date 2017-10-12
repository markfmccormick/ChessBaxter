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

# Instantiate CvBridge
bridge = CvBridge()
c = 0

def image_callback(msg):
    
    print("\nReceived an image!")
    try:
        # Convert your ROS Image message to OpenCV2
        cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
        # cv2.imshow('cv_img', cv2_img)			THE NEW OPENED WINDOW DOESN'T WORK
    except CvBridgeError, e:
        print(e)
    else:
        # Save your OpenCV2 image as a jpeg 
        global c
        if(c == 100):
            c = 0
        c  += 1
        cv2.imwrite('kinect_images/camera_image' + str(c) + '.jpeg', cv2_img)
        time.sleep(1)

def main():
    rospy.init_node('image_listener')
    # Define your image topic
    image_topic = "/kinect2/hd/image_color"
    # Set up your publisher - DOES NOT DO ANYTHING FOR NOW AS I NEED TO RUN roslaunch kinect2_bridge kinect2_bridge.launch
    pub = rospy.Publisher(image_topic, Image, queue_size=0)
    # Set up your subscriber and define its callback

    rospy.Subscriber(image_topic, Image, image_callback)
    # Spin until ctrl + c
    rospy.spin()

if __name__ == '__main__':
    main()
