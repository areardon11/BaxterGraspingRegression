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

points = None

class pc_gener(object):
	def __init__(self, delta=0.15):
		self.listener = tf.TransformListener()
		t = None
		while not t:
			try:
				t, r = self.listener.lookupTransform('/base', '/ar_marker_1', rospy.Time(0))
			except:
				continue
		x_b = t[0] - delta, t[0] + delta
		y_b = t[1] - delta, t[1] + delta
		z_b= t[2] + .02, t[2] + 3*delta
		self.bounds = [x_b, y_b, z_b]
		self.pc = None
		self.points = None

	def transform_pc(self, pc, trans, rot):
		t_points = []
		G = quaternion_matrix(rot)
		G[:3,3] += trans
		for p in pc:
			p = np.append(p, 1)
			t_points.append(G.dot(p)[:3])
		return np.vstack(t_points)

	def box_pc(self, pc):
	    potentials = []
	    for i in range(3):
	        pl = np.where(pc[:,i] >= self.bounds[i][0])[0]
	        ph = np.where(pc[:,i] <= self.bounds[i][1])[0]
	        p = np.intersect1d(pl, ph)
	        potentials.append(p)
	    p = np.intersect1d(potentials[0], potentials[1])
	    p = np.intersect1d(p, potentials[2])
	    return pc[p]

	def pc_callback(self, msg):
		self.points = np.array(list(pc2.read_points(msg, skip_nans=True)))

	def get_one_pc(self, kinect):
		topic = '/' + kinect + '/depth_registered/points'
		klink = '/' + kinect + '_link'
		sub = rospy.Subscriber(topic, PointCloud2, self.pc_callback)
		rospy.sleep(2.0)
		sub.unregister()
		t,r = None, None
		done = False
		while not done:
			try:
				t, r = self.listener.lookupTransform('/base', klink , rospy.Time(0))
				done = True
			except:
				continue
		return self.transform_pc(self.points[:,:3], t,r)

	def gen_new_pc(self):
		print('Starting')
		pc1 = self.get_one_pc('kinect1')
		print('Done PC1')
		pc2 = self.get_one_pc('kinect2')
		print('Done PC2')
		self.pc = self.box_pc(np.vstack((pc1,pc2)))

	def get_pc(self):
		return self.pc