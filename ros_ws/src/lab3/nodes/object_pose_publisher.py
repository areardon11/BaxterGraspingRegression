#!/usr/bin/env python  
# import roslib
# roslib.load_manifest('lab3')
# import rospy
# import math
# import numpy as np
# import tf
# import geometry_msgs.msg

# if __name__ == '__main__':
#     rospy.init_node('object_pose_publisher')

#     broadcaster = tf.TransformBroadcaster()
 
#     R_ar_obj = np.eye(3)
#     t_ar_obj = np.array([-0.094, -0.074, 0.106])
#     T_ar_obj = np.eye(4)
#     T_ar_obj[:3,:3] = R_ar_obj
#     T_ar_obj[:3, 3] = t_ar_obj
#     q_ar_obj = tf.transformations.quaternion_from_matrix(T_ar_obj)
    
#     print 'Publishing object pose'
    
#     rate = rospy.Rate(1.0)
#     while not rospy.is_shutdown():
#         try:
#             broadcaster.sendTransform(t_ar_obj, q_ar_obj, rospy.Time.now(), '/graspable_object', '/ar_marker_8')
#         except:
#             continue
#         rate.sleep()
import roslib
roslib.load_manifest('lab3')
import rospy
import numpy as np
import tf
from os import system
import subprocess

if __name__ == '__main__':
    rospy.init_node('object_pose_publisher')
    proc = subprocess.Popen(['rosrun','ar_track_alvar','individualMarkersNoKinect','marker_size:=4.8','max_new_marker_error:=0.08','max_track_error:=0.2','cam_image_topic:=/cameras/right_hand_camera/image','cam_info_topic:=/cameras/right_hand_camera/camera_info','output_frame:=/right_hand_camera'])
    
    listener = tf.TransformListener()
    count = 0
    t_list, r_list = [], []
    while True:
        try:
            t,r = listener.lookupTransform('right_hand_camera', 'ar_marker_1', rospy.Time(0))
            if t is not None and r is not None:
                t_list.append(np.array(t))
                r_list.append(np.array(r))
                count += 1
                if count == 5:
                    break
        except:
            continue

    avg_t = tuple(sum(t_list) / float(len(t_list)))
    avg_r = tuple(sum(r_list) / float(len(r_list)))

    print 'The translation from Base to AR Marker 1 is: ' + str(avg_t)
    print 'The rotation from Base to AR Marker 1 is: ' + str(avg_r) 

    broadcaster = tf.TransformBroadcaster()
    rate = rospy.Rate(1.0)
    rospy.sleep(5)
    print('Here1')
    proc.kill()
    #system('rosnode kill ar_track_alvar')
    while not rospy.is_shutdown():
        broadcaster.sendTransform(avg_t, 
                                  avg_r, 
                                  rospy.Time.now(), 
                                  '/ar_marker_1', 
                                  '/base')
        rate.sleep()    