#!/usr/bin/env python
import time
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
import numpy as np
import cv2
import sys
from collections import namedtuple
import math
from math import floor, atan2, pi, cos, sin, sqrt
from cv_bridge import CvBridge, CvBridgeError

def callback(data):

    bridge=CvBridge()
    #convert images to opencv image
    try:
        cv_image = bridge.imgmsg_to_cv2(data, "8UC3")
    except CvBridgeError as e:
        print(e)
        
    #show image
    cv2.namedWindow("Image")
    if (not cv_image is None):
        cv2.imshow("Image",cv_image)
    if cv2.waitKey(1)!=-1:     #Burak, needs to modify this line to work on your computer, THANKS!
        cv2.destroyAllWindows()

if __name__ == '__main__': 
    rospy.init_node('cam_stream_node', anonymous=False)
    sub = rospy.Subscriber("/image_raw",Image,callback)
    rospy.spin()
