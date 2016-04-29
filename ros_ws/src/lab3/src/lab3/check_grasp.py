#!/usr/bin/env python
import sys
import rospy
import numpy as np
from goemetry_msgs.msg import PoseStamped
from baxter_interface import gripper as baxter_gripper
from baxter_core_msgs.msg import EndEffectorState 

BASE_FRAME = 'base'
HEIGHT_OFFSET = 2.0
OPEN_AMOUNT = 100.0
hld_frc = 1
mv_frc = 25
grasped = False

def grasp():
    right_gripper.close(block=True)

def release():
    right_gripper.command_position(OPEN_AMOUNT, block=True)

def check_grasp(msg):
    global grasped
    if msg.gripping == 1:
        grasped = True
    else:
        grasped = False

def loop():
    while not ropsy.is_shutdown():
        raw_input('Press <Enter> to CLOSE the grippers:')
        grasp()
        rospy.sleep(0.5)
        raw_input('Press <Enter> to CHECK GRASP:')
        print('Grasped? = ', grasped)
        raw_input('Press <Enter> to OPEN grippers:')
        release()

if __name__ == "__main__":
    rospy.init_node("check_grasp")
    
    right_gripper = baxter_gripper.Gripper('right')
    right_gripper.calibrate()
    #right_gripper.set_holding_force(hld_frc)
    #right_gripper.set_moving_force(mv_frc)

    listener = tf.TransformListener()

    rospy.Subscriber('robot/end_effector/right_gripper/state', EndEffectorState, check_grasp)

    rospy.spin()