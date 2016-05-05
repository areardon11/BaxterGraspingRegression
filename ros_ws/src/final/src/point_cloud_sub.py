#!/usr/bin/env python
import roslib
import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import pickle
import tf
from geometry_msgs.msg import TransformStamped

count1 = 0
count2 = 0

def callback(msg):
	global count1
	if count1 == 0:
		points = list(pc2.read_points(msg, skip_nans=True))
		pickle.dump(points, open('kinect1_pc2_read', 'wb'))
		pickle.dump(msg, open('kinect1_pc2', 'wb'))
		count1 += 1
		print 'Done1'

def callback2(msg):
	global count2
	if count2 == 0:
		points = list(pc2.read_points(msg, skip_nans=True))
		pickle.dump(points, open('kinect2_pc2_read', 'wb'))
		pickle.dump(msg, open('kinect2_pc2', 'wb'))
		count2 += 1
		print 'Done2'


if __name__ =='__main__':
    rospy.init_node('kinect_depth',anonymous=True)
    rospy.Subscriber('/kinect1/depth_registered/points', PointCloud2, callback)
    rospy.Subscriber('/kinect2/depth_registered/points', PointCloud2, callback2)

    listener = tf.TransformListener()
    done = False
    tf_dict = {}
    while not done:
        try:
			tf_dict['kinect1_base'] = listener.lookupTransform('/base', '/kinect1_link',\
                 rospy.Time(0))
			tf_dict['kinect2_base'] = listener.lookupTransform('/base', '/kinect2_link',\
                 rospy.Time(0))
			tf_dict['ar1_base'] = listener.lookupTransform('/base', '/ar_marker_1',\
                 rospy.Time(0))
			tf_dict['base_kinect1'] = listener.lookupTransform('/kinect1_link', '/base',\
                 rospy.Time(0))
			tf_dict['base_kinect2'] = listener.lookupTransform('/kinect2_link', '/base',\
                 rospy.Time(0))
			tf_dict['base_ar1'] = listener.lookupTransform('/ar_marker_1', '/base',\
                 rospy.Time(0))
			tf_dict['kinect1_ar1'] = listener.lookupTransform('/ar_marker_1', '/kinect1_link',\
                 rospy.Time(0))
			tf_dict['kinect2_ar1'] = listener.lookupTransform('/ar_marker_1', '/kinect2_link',\
                 rospy.Time(0))
			tf_dict['ar1_kinect1'] = listener.lookupTransform('/kinect1_link', '/ar_marker_1',\
                 rospy.Time(0))
			tf_dict['ar1_kinect2'] = listener.lookupTransform('/kinect2_link', '/ar_marker_1',\
                 rospy.Time(0))
			done = True
        except:
            continue
    
    pickle.dump(tf_dict, open('tf_dict', 'wb'))
    print("Done3")
    rospy.spin()
