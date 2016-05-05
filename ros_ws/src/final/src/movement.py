#!/usr/bin/env python
import sys
import rospy
import moveit_commander
import tf
from tf.transformations import *
import numpy as np
from geometry_msgs.msg import PoseStamped
from baxter_interface import gripper as baxter_gripper
import baxter_interface
from moveit_msgs.msg import *

# the name of the world frame
BASE_FRAME = 'base'
HEIGHT_OFFSET = 1.0
OPEN_AMOUNT = 100.0
hld_frc = 1
mv_frc = 25

def lookup_transform(name):
    while True:
        try:
            trans, rot = listener.lookupTransform(BASE_FRAME, name, rospy.Time(0))
            break
        except:
            continue
    return list(trans), list(rot)

def assign_xyz(arr, xyz):
    xyz.x = arr[0]
    xyz.y = arr[1]
    xyz.z = arr[2]
    if hasattr(xyz, 'w'):
        xyz.w = arr[3]
    return xyz

def assign_arr(xyz, arr=None):
    has_w = hasattr(xyz, 'w')
    if arr is None:
        arr = np.zeros(4 if has_w else 3)
    arr[0] = xyz.x
    arr[1] = xyz.y
    arr[2] = xyz.z
    if has_w: arr[3] = xyz.w
    return arr

def goto(trans, rot=[.504, .544, -.471, .478]):
    planner = left_planner

    goal = PoseStamped()
    goal.header.frame_id = BASE_FRAME

    assign_xyz(trans, goal.pose.position)
    assign_xyz(rot, goal.pose.orientation)

    # find a plan to get there
    planner.set_pose_target(goal)
    planner.set_start_state_to_current_state()
    plan = planner.plan()
    print('Done Planning')
    planner.execute(plan)
    rospy.sleep(0.5)
    print('Done Executing')
    
def goto_imag_pose():
    goto(imaging_pose[0], imaging_pose[1])
    rospy.sleep(0.5)

def goto_pregrasp_pose():
    goto(pregrasp_pose[0], pregrasp_pose[1])
    rospy.sleep(0.5)

def lift():
    left_goal = lookup_transform('left_hand')
    left_goal = list(left_goal)
    left_goal[0][1] += .5
    goto(left_goal[0], left_goal[1])
    rospy.sleep(0.5)

def grasp():
    left_gripper.close(block=True)
    rospy.sleep(.5)

def release():
    left_gripper.command_position(OPEN_AMOUNT, block=True)
    rospy.sleep(.5)

if __name__ == "__main__":
    moveit_commander.roscpp_initialize(sys.argv)
    
    rospy.init_node('grasp')
    
    left_arm = baxter_interface.Limb('left')
    left_planner = moveit_commander.MoveGroupCommander('left_arm')
    left_planner.set_planner_id('RRTConnectkConfigDefault')
    left_planner.set_planning_time(10)
    left_gripper = baxter_gripper.Gripper('left')

    listener = tf.TransformListener()
    rospy.spin()

    # goto(pre_trans, pre_rot)
    # release()
    # grasp()
    # goto((trans[0], trans[1], trans[2] + .2), rot)

    # rospy.sleep(.5)

    # goto(trans, rot)
    # release()
    # goto(pre_trans, pre_rot)
    
