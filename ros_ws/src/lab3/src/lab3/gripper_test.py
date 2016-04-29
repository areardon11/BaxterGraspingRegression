#!/usr/bin/env python
import rospy
from baxter_interface import gripper as baxter_gripper

rospy.init_node('gripper_test')

#Set up the left gripper
left_gripper = baxter_gripper.Gripper('left')

#Calibrate the gripper (other commands won't work unless you do this first)
print('Calibrating...')
left_gripper.calibrate()
rospy.sleep(2.0)

#Close the left gripper
print('Closing...')
left_gripper.close(block=True)
rospy.sleep(1.0)

#Open the left gripper
print('Opening...')
left_gripper.open(block=True)
rospy.sleep(1.0)
print('Done!')