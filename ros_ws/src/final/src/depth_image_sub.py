#!/usr/bin/env python
import roslib
import rospy
from sensor_msgs.msg import Image
import sensor_msgs.point_cloud2 as pc2
import pickle
import tf
from geometry_msgs.msg import TransformStamped
from IPython import embed
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import matplotlib.pyplot as plt

bridge = CvBridge()
imgList = []
first = True
def callback(msg):
	global imgList, first
	try:
		cv_image = bridge.imgmsg_to_cv2(msg, 'passthrough')
		imgList.append(cv_image)
	except CvBridgeError as e:
		print(e)

	if len(imgList) == 10 and first:
		first = False
		imgList = []

	if len(imgList) == 50:
		s = np.sum(imgList, axis=0) / len(imgList)
		plt.imshow(s, cmap='gray')
		# plt.show()
		embed()
		# cv2.imshow('Image Window', cv_image)
		# cv2.waitKey(3)
		imgList = []

if __name__ =='__main__':
    rospy.init_node('kinect_depth',anonymous=True)
    rospy.Subscriber('/kinect1/depth_registered/image_raw', Image, callback)
    print("Done3")
    rospy.spin()
