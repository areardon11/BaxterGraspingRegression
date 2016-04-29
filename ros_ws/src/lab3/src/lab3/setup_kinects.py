#!/usr/bin/env python
import sys
import rospy
import tf
from tf.transformations import *
import numpy as np
from geometry_msgs.msg import PoseStamped
import baxter_interface





if __name__ == '__main__':
	rospy.init_node("kinect_setup")
	rate = rospy.Rate(5)
	listener = tf.TransformListener()
	while not rospy.is_shutdown():
		try:
			t,r = listener.lookupTransform('base', 'ar_marker_0', rospy.Time(0))
			print "Transform: ",t
			print "Rotation: ",r
		except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
			continue
		rate.sleep()
	rospy.spin()