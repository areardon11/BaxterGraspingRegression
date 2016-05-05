#!/usr/bin/env python
import roslib
roslib.load_manifest('final')
import rospy
import numpy as np
import tf
from os import system
import roslaunch
import tf2_ros

MARKER_SIZE = "13.35"
MAX_NEW_MARKER_ERROR = "0.08"
MAX_TRACK_ERROR = "0.2"
CAM_IMAGE_TOPIC = "/cameras/right_hand_camera/image"
CAM_INFO_TOPIC = "/cameras/right_hand_camera/camera_info"
OUTPUT_FRAME = "/right_hand_camera"
MNAME = "baxter_right_hand_camera"

def string_args():
    return ' '.join((MARKER_SIZE, MAX_NEW_MARKER_ERROR, MAX_TRACK_ERROR,\
                CAM_IMAGE_TOPIC, CAM_INFO_TOPIC, OUTPUT_FRAME))

def update_args(name):
    global CAM_IMAGE_TOPIC, CAM_INFO_TOPIC, OUTPUT_FRAME, MNAME
    CAM_IMAGE_TOPIC = "/"+name+"/rgb/image_color"
    CAM_INFO_TOPIC = "/"+name+"/rgb/camera_info"
    OUTPUT_FRAME = "/"+name+"_link"
    MNAME = name
    
if __name__ == '__main__':
    rospy.init_node('right_hand_camera_publisher')
    
    listener = tf.TransformListener()
    count = 0
    t_list, r_list = [], []

    package = 'ar_track_alvar'
    executable = 'individualMarkersNoKinect'
    args = string_args()
    node = roslaunch.core.Node(package, executable, args=args, respawn=False)
    
    launch = roslaunch.scriptapi.ROSLaunch()
    launch.start()

    process = launch.launch(node)
    while True:
        try:
            t,r = listener.lookupTransform('base', 'ar_marker_1', rospy.Time(0))
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

    print 'The translation from Base to AR Marker 1 is: ' + str(avg_t)
    print 'The rotation from Base to AR Marker 1 is: ' + str(avg_r) 
    process.stop()
    
    rate = rospy.Rate(20.0)
    br = tf.TransformBroadcaster()
    while not rospy.is_shutdown():
        br.sendTransform(avg_t, avg_r, rospy.Time.now(), 'ar_marker_1', 'base')
        rate.sleep()
