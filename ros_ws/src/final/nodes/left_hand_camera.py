#!/usr/bin/env python
import roslib
roslib.load_manifest('final')
import rospy
import numpy as np
import tf
from os import system

if __name__ == '__main__':
    rospy.init_node('left_hand_camera_publisher')
    
    listener = tf.TransformListener()
    count = 0
    t_list, r_list = [], []

    while True:
        try:
            t,r = listener.lookupTransform('left_hand_camera', 'ar_marker_1', rospy.Time(0))
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

    broadcaster = tf.TransformBroadcaster()
    rate = rospy.Rate(1.0)

    system('rosnode kill ar_track_alvar_left_hand')

    while not rospy.is_shutdown():
        broadcaster.sendTransform(avg_t, 
                                  avg_r, 
                                  rospy.Time.now(), 
                                  '/ar_marker_1', 
                                  '/base')
        rate.sleep()    