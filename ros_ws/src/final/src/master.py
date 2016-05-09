#!/usr/bin/env python
import roslib
import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import pickle
import tf
from tf.transformations import *
from geometry_msgs.msg import TransformStamped
import numpy as np
from std_msgs import msg
from IPython import embed
from pc_generator import pc_gener

def visualize_pc(pc):
	pcl_pub = rospy.Publisher("/custom_pc", PointCloud2, queue_size=10)
	points = pc.tolist()
	rate = rospy.Rate(1)
	while not rospy.is_shutdown():
	    header = msg.Header()
	    header.stamp = rospy.Time.now()
	    header.frame_id = 'base' 
	    point_cloud = pc2.create_cloud_xyz32(header, points)
	    pcl_pub.publish(point_cloud)
	    rate.sleep()

if __name__ == '__main__':
	rospy.init_node('master', anonymous=True)
	pc_gen = pc_gener()

