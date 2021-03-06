import numpy as np 
import tf
from tf.transformations import *
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from std_msgs import msg
import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
import matplotlib.pyplot as plt

"""The steps for collecting depth image and then converting to a merged
point cloud in Baxter's base frame:

1. Collect an averaged depth image from each Kinect. Call these dimg1 and dimg2
for Kinect1 and Kinect2 respectively. (This should be done in a separate method).

2. Compute the tfs for each Kinect to the base, i.e. FROM KinectX TO base -->
tf.lookupTransform('base', /KinectX_link, rospy.time(0)). Store these tfs as
transX, rotX for the respective KinectX.

3. Call merge_dimg_to_pcs(dimg1, dimg2, trans1, rot1, trans2, rot2) to get 
a merged pointcloud in Baxter's base frame. The merged pointcloud
will be an nX3 numpy array.

4. (Optional) Visualize the pointcloud in RVIZ using rviz_pc_visualizer() to double 
check correctness.

5. Run the merged pointcloud through the bounding box by calling boxer() with 
the desired arguments.

6. (Optional) Visualize this new bounded pointcloud in RVIZ using rviz_pc_visualizer().

7. Send this boxed pointcoud to sampler."""

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

def boxer(pc, bounds):
    potentials = []
    for i in range(3):
        print(i)
        pl = np.where(pc[:,i] >= bounds[0][0])[0]
        ph = np.where(pc[:,i] <= bounds[0][1])[0]
        p = np.intersect1d(pl, ph)
        potentials.append(p)
    p = np.intersect1d(potentials[0], potentials[1])
    p = np.intersect1d(p, potentials[2])
    return pc[p]

def merge_dimg_to_pcs(dimg1, dimg2, trans1, rot1, trans2, rot2):
	h1, w1 = dimg1.shape
	h2, w2 = dimg2.shape
	K1 = np.array([[525,0,w1/2],[0,525,h1/2],[0,0,1]], dtype=np.uint16)
	K2 = np.array([[525,0,w2/2],[0,525,h2/2],[0,0,1]], dtype=np.uint16) # K1 should equal K2 when the depth images are of the same size
	pc1 = dimg_to_pc(dimg1, K1, trans1, rot1)
	pc2 = dimg_to_pc(dimg2, K2, trans2, rot2)
	return np.array(pc1 + pc2)

def rviz_pc_visualizer(pc):
	rospy.init_node('pc_publisher', anonymous=True)
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