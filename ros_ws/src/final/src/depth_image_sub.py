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

class dimg_collecter:

	def __init__(self, kinect, num_to_collect):
		# node_name = kinect + '_depth'
		# rospy.init_node(node_name,anonymous=True)
		self.kinect = kinect
		self.num_to_collect = num_to_collect
		self.img_history = []
		self.bridge = CvBridge()
		topic = '/' + kinect + '/depth_registered/image_raw'
		self.avg_dimg = None

	def collect_dimg(self, img):
		try:
			cv_img = self.bridge.imgmsg_to_cv2(img, 'passthrough')
		except CvBridgeError as e:
			print(e)

		if len(self.img_history) < self.num_to_collect:
			self.img_history.append(cv_img)
		
	def collect_avg_dimg(self):
		self.image_sub = rospy.Subscriber(self.topic, Image, self.collect_dimg)
		rospy.sleep(3.0)
		while len(self.img_history) < self.num_to_collect: continue
		self.image_sub.unregister()
		self.avg_dimg = np.mean(self.img_history, axis=0, dtype=np.uint8)
		return self.avg_dimg

	def show_avg_img(self):
		cv2.imshow('Average Depth Image', self.avg_dimg)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

def main():
	rospy.init_node('dimg_collecter', anonymous=True)
	collector = dimg_collecter('kinect1', 10)
	raw_input('Press <Enter> to collect an averaged depth image for Kinect1:')
	collector.collect_avg_dimg()
	collector.show_avg_img()
	rospy.spin()

if __name__ == '__main__':
	main()

# bridge = CvBridge()
# imgList = []
# first = True
# def callback(msg):
# 	global imgList, first
# 	try:
# 		cv_image = bridge.imgmsg_to_cv2(msg, 'passthrough')
# 		imgList.append(cv_image)
# 	except CvBridgeError as e:
# 		print(e)

# 	if len(imgList) == 10 and first:
# 		first = False
# 		imgList = []

# 	if len(imgList) == 50:
# 		s = np.sum(imgList, axis=0) / len(imgList)
# 		plt.imshow(s, cmap='gray')
# 		# plt.show()
# 		embed()
# 		# cv2.imshow('Image Window', cv_image)
# 		# cv2.waitKey(3)
# 		imgList = []

# if __name__ =='__main__':
#     rospy.init_node('kinect_depth',anonymous=True)
#     rospy.Subscriber('/kinect1/depth_registered/image_raw', Image, callback)
#     print("Done3")
#     rospy.spin()

