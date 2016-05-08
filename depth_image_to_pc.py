import numpy as np 
import tf
from tf.transformations import *

def dimg_to_pc(dimg, K, trans, rot):
	pc = []
	h, w = dimg.shape
	t_cam_to_base = trans
	R_cam_to_base = quaternion_matrix(rot)
	K_inv = np.linalg.inv(K)

	for i in range(w):
		for j in range(h):
			d = dimg[j,i]
			u = [i,j,1]
			x_cam = d * np.dot(K_inv, u)
			x_base = np.dot(R_cam_to_base, x_cam) + t_cam_to_base
			pc.append(x_base)
	return pc

def merge_dimg_to_pcs(dimg1, dimg2, trans1, rot1, trans2, rot2):
	h1, w1 = dimg1.shape
	h2, w2 = dimg2.shape
	K1 = np.array([[525,0,w1/2],[0,525,h1/2],[0,0,1]], dtype=np.uint16)
	K2 = np.array([[525,0,w2/2],[0,525,h2/2],[0,0,1]], dtype=np.uint16) # K1 should equal K2 when the depthm images are of the same size
	pc1 = dimg_to_pc(dimg1, K1, trans1, rot1)
	pc2 = dimg_to_pc(dimg2, K2, trans2, rot2)
	return pc1 + pc2

def tester():
	""" The steps for collecting depth image and then converting to a merged
	point cloud in Baxter's base frame:

	1. Collect an averaged depth image from each Kinect. Call these dimg1 and dimg2
	for Kinect1 and Kinect2 respectively. (This should be done in a separate method).


	2. Compute the tfs for each Kinect to the base, i.e. FROM KinectX TO base -->
	tf.lookupTransform('base', /KinectX_link, rospy.time(0)). Store these tfs as
	transX, rotX for the respective KinectX.

	3. Call merge_dimg_to_pcs(dimg1, dimg2, trans1, rot1, trans2, rot2) to get 
	a merged pointcloud in Baxter's base frame. The merged pointcloud
	will be a Python List of (1x3) numpy arrays.

	4. (Optional) Develop a method to visualize/publish the pointcloud to double 
	check correctness.

	5. Send this pointcoud to featurizer.""""