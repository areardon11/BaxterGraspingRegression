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
def transform_pc(pc, trans, rot):
	t_points = []
	G = quaternion_matrix(rot)
	R_cam_to_base = quaternion_matrix(rot)[:3,:3]
	G[:3,3] += trans
	for p in pc:
		#p = np.append(p, 1)
		a = np.dot(R_cam_to_base, p) + trans
		t_points.append(a)
		#t_points.append(G.dot(p)[:3])
	return np.vstack(t_points)

def boxer(pc, bounds):
    potentials = []
    for i in range(3):
        pl = np.where(pc[:,i] >= bounds[i][0])[0]
        ph = np.where(pc[:,i] <= bounds[i][1])[0]
        p = np.intersect1d(pl, ph)
        potentials.append(p)
    p = np.intersect1d(potentials[0], potentials[1])
    p = np.intersect1d(p, potentials[2])
    return pc[p]

def box_and_display(pc):
	t = None
	delta = .15
	while not t:
		try:
			t, r = listener.lookupTransform('/base', '/ar_marker_1', rospy.Time(0))
		except:
			continue
	x_b = t[0] - delta, t[0] + delta
	y_b = t[1] - delta, t[1] + delta
	z_b= t[2] + .02, t[2] + 3*delta 
	bounds = [x_b, y_b, z_b]
	boxed_pc = boxer(pc, bounds)
	with open('boxed', 'wb') as f:
		pickle.dump(boxed_pc, f)
	visualize_pc(boxed_pc)

def pc_callback(msg):
	global points
	points = np.array(list(pc2.read_points(msg, skip_nans=True)))

def get_pc(kinect):
	global points
	topic = '/' + kinect + '/depth_registered/points'
	klink = '/' + kinect + '_link'
	sub = rospy.Subscriber(topic, PointCloud2, pc_callback)
	rospy.sleep(2.0)
	sub.unregister()

	t,r = None, None
	done = False
	while not done:
		try:
			t, r = listener.lookupTransform('/base', klink , rospy.Time(0))
			done = True
		except:
			continue

	return transform_pc(points[:,:3], t,r)

def pc_controller():
    pc1 = get_pc('kinect1')
    pc2 = get_pc('kinect2')
    pc = np.vstack((pc1,pc2))
    print('Merged PC')
    box_and_display(pc)
	
if __name__ =='__main__':
    rospy.init_node('kinect_depth',anonymous=True)
    # rospy.Subscriber('/kinect1/depth_registered/points', PointCloud2, callback)
    # rospy.Subscriber('/kinect2/depth_registered/points', PointCloud2, callback2)
    listener = tf.TransformListener()
    pc_controller()
