#!/usr/bin/env python
import roslib
roslib.load_manifest('final')
import rospy
import numpy as np
import tf
import roslaunch
import tf2_ros
from geometry_msgs.msg import TransformStamped

name = 'kinect1'
MARKER_SIZE = "13.35"
MAX_NEW_MARKER_ERROR = "0.08"
MAX_TRACK_ERROR = "0.2"
CAM_IMAGE_TOPIC = "/"+name+"/rgb/image_color"
CAM_INFO_TOPIC = "/"+name+"/rgb/camera_info"
OUTPUT_FRAME = "/"+name+"_link"
MNAME = name



if __name__ == '__main__':
    rospy.init_node('kinect1_location_broadcaster')
    
    listener = tf.TransformListener()
    count = 0
    t_list, r_list = [], []
    
    package = 'ar_track_alvar'
    executable = 'individualMarkersNoKinect'
    args = ' '.join((MARKER_SIZE, MAX_NEW_MARKER_ERROR, MAX_TRACK_ERROR,\
                    CAM_IMAGE_TOPIC, CAM_INFO_TOPIC, OUTPUT_FRAME))

    node = roslaunch.core.Node(package, executable, args=args, respawn=False)

    launch = roslaunch.scriptapi.ROSLaunch()
    launch.start()

    process = launch.launch(node)
    while True:
        print("ahhhhh")
        try:
            t,r = listener.lookupTransform('ar_marker_1', 'kinect1_link', rospy.Time(0))
            if t is not None and r is not None:
                t_list.append(np.array(t))
                r_list.append(np.array(r))
                count += 1
                if count == 5:
                    break
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue

    avg_t = tuple(sum(t_list) / float(len(t_list)))
    avg_r = tuple(sum(r_list) / float(len(r_list)))

    print 'The translation from Kinect 1 to AR Marker 1 is: ' + str(avg_t)
    print 'The rotation from Kinect 1 to AR Marker 1 is: ' + str(avg_r) 
    process.stop()

    broadcaster = tf.TransformBroadcaster()
    rate = rospy.Rate(20.0)

    while not rospy.is_shutdown():
        broadcaster.sendTransform(avg_t, 
                                  avg_r, 
                                  rospy.Time.now(), 
                                  '/kinect1_link', 
                                  '/ar_marker_1')
        rate.sleep()    
