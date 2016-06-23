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

BASE_FRAME = 'base'
HEIGHT_OFFSET = [0.0, 0.0, 0.3]
GOAL_OFFSET = [-0.10, 0.0, 0.0]

def lookup_transform(name):
    while True:
        try:
            trans, rot = listener.lookupTransform(BASE_FRAME, name, rospy.Time(0))
            break
        except:
            continue
    return [list(trans), list(rot)]

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

def goto(trans, rot):
    planner = left_planner

    goal = PoseStamped()
    goal.header.frame_id = BASE_FRAME

    assign_xyz(trans, goal.pose.position)
    assign_xyz(rot, goal.pose.orientation)

    # find a plan to get there
    planner.set_pose_target(goal)
    planner.set_start_state_to_current_state()
    plan = planner.plan()
    planner.execute(plan)
    rospy.sleep(1.5)
    
def lift(goal_pose):
    lift_goal = goal_pose
    lift_goal[0] += HEIGHT_OFFSET
    goto(lift_goal[0], lift_goal[1])

def grasp():
    left_gripper.close(block=True)
    rospy.sleep(.5)

def release():
    left_gripper.open(block=True)
    #left_gripper.command_position(OPEN_AMOUNT, block=True)
    rospy.sleep(.5)

def execute_grasp(trans, rot):
    goal_pose = [trans, rot]
    pregrasp_pose = goal_pose
    pregrasp_pose[0] += GOAL_OFFSET
    print 'Pregrasp Pose:', pregrasp_pose
    goto(pregrasp_pose[0], pregrasp_pose[1])
    print 'Goal Pose:', goal_pose
    goto(goal_pose[0], goal_pose[1])
    grasp()
    lift(goal_pose)
    raw_input('Press Enter to put object down.')
    goto(goal_pose[0], goal_pose[1])
    release()
    goto(pregrasp_pose[0], pregrasp_pose[1])
    print('Done executing grasp.')

