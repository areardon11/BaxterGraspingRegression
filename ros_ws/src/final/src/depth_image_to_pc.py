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
from depth_image_sub import dimg_collector

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

dimg = None
pc_data = None

def dimg_filter(x,y,z):
	data = np.vstack((x.ravel(),y.ravel(),z.ravel())).T
	nan = np.isnan(data)
	idx = np.where(nan[:,2] != True)[0]
	data = data[idx]
	return data

def dimg_to_pc(dimg, K, trans, rot):
	cx, cy = K[0,2], K[1,2]
	fx, fy = K[0,0], K[1,1]
	rows, cols = dimg.shape
	c,r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
	valid = (dimg > 0) & (dimg < 255)
	z = np.where(valid, dimg/256.0, np.nan)
	x = np.where(valid, z * (c - cx) / fx, 0)
	y = np.where(valid, z * (r - cy) / fy, 0)
	return x,y,z
	# pc = []
	# h, w = dimg.shape
	# t_cam_to_base = trans
	# R_cam_to_base = quaternion_matrix(rot)[:3,:3]
	# K_inv = np.linalg.inv(K)

	# for i in range(w):
	# 	for j in range(h):
	# 		d = dimg[j,i]
	# 		u = [i,j,1]
	# 		x_cam = d * np.dot(K_inv, u)
	# 		x_base = np.dot(R_cam_to_base, x_cam) + t_cam_to_base
	# 		pc.append(x_base)
	# return pc

def transform_pc(pc, trans, rot):
	t_points = []
	G = quaternion_matrix(rot)
	G[:3,3] += trans
	for p in pc:
		p = np.append(p, 1)
		t_points.append(G.dot(p)[:3])
	return np.vstack(t_points)   

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
	#rospy.init_node('pc_publisher', anonymous=True)
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

def main():
	global dimg, pc_data
	k1_collector = dimg_collector('kinect1',10)
	raw_input('Press <Enter> to collect an averaged depth image for Kinect1:')
	dimg = k1_collector.collect_avg_dimg()
	x,y,z = dimg_to_pc(dimg, K, 0,0)
	pc_data = dimg_filter(x,y,z)
	#k1_collector.show_avg_img()
	#rospy.spin()

if __name__ == '__main__':
	rospy.init_node('blah2', anonymous=True)
	listener = tf.TransformListener()
	K = np.array([[525,0,640/2],[0,525,480/2],[0,0,1]], dtype=np.uint16)
	main()