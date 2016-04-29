#!/usr/bin/env python
import sys
import rospy
import moveit_commander
import tf
import obj_file
import numpy as np
from goemetry_msgs.msg import PoseStamped
from baxter_interface import gripper as baxter_gripper
from moveit_msgs.msg import *
from fc import *

# the name of the world frame
BASE_FRAME = 'base'
HEIGHT_OFFSET = 2.0
OPEN_AMOUNT = 100.0
hld_frc = 1
mv_frc = 25

def get_pose(arm):
    pose = arm.endpoint_pose()
    pos = pose['position']
    Q = pose['orientation']
    return pos.x, pos.y, pos.z, [Q.x, Q.y, Q.z, Q.w]

def lookup_transform(name):
    trans, rot = listener.lookupTransform(BASE_FRAME, name, rospy.Time(0))
    x, y, z, rott = get_pose(name)
    trans = np.array(trans) + np.array([x,y,z])
    rot = np.array(rot) + np.array(rott)
    return trans/2, rot/2

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
    planner = right_planner

    goal = PoseStamped()
    goal.header.frame_id = BASE_FRAME

    assign_xyz(rot, goal.pose.orientation)

    # find a plan to get there
    planner.set_pose_target(goal)
    planner.set_start_state_to_current_state()
    plan = planner.plan()

    # go there
    planner.execute(plan)
    rospy.sleep(0.5)

    assign_xyz(trans, goal.pose.position)
    planner.set_pose_target(goal)
    planner.set_start_state_to_current_state()
    plan = planner.plan()

    # go there
    planner.execute(plan)
    rospy.sleep(0.5)

def goto_imag_pose():
    right_goal = imaging_pose
    goto(right_goal[0], right_goal[1])
    rospy.sleep(0.5)

def goto_pregrasp_pose():
    right_goal = pregrasp_pose
    goto(right_goal[0], right_goal[1])
    rospy.sleep(0.5)

def lift():
    right_goal = lookup_transform('right_hand')
    right_goal.pose.position.z += HEIGHT_OFFSET
    goto(right_goal[0], right_goal[1])
    rospy.sleep(0.5)

def grasp():
    right_gripper.close(block=True)

def release():
    right_gripper.command_position(OPEN_AMOUNT, block=True)

def check_grasp():
    return 1

def calib():
    global ar_loc, imaging_pose, pregrasp_pose

    ar_loc = None
    imaging_pose = None
    pregrasp_pose = None

    raw_input('Press <Enter> to save location of AR Tag: Do NOT move the arm.')
    while ar_loc == None:
        try:
            ar_loc= listener.lookupTransform('base', 'right_gripper', rospy.Time(0))
            #add offset to z portion of ar_loc
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue
    print("The location of the AR Tag has been saved!")

    raw_input('Press <Enter> to save location of Imaging Pose.')
    while imaging_pose == None:
        try:
            imaging_pose = lookup_transform('right_hand')
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue
    print("Imaging Pose has been recorded")

    raw_input('Press <Enter> to save location of Imaging Pose.')
    while pregrasp_pose == None:
        try:
            pregrasp_pose = lookup_transform('right_hand')
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue
    print("Imaging Pose has been recorded")


if __name__ == "__main__":
    moveit_commander.roscpp_initialize(sys.argv)
    
    rospy.init_node("grasping")
    
    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()
    right_arm = moveit_commander.MoveGroupCommander('right_arm')
    right_arm.set_planner_id('RRTConnectkConfigDefault')
    right_arm.set_planning_time(10)
    
    right_gripper = baxter_gripper.Gripper('right')
    right_gripper.calibrate()
    right_gripper.set_holding_force(hld_frc)
    right_gripper.set_moving_force(mv_frc)

    listener = tf.TransformListener()

    #rospy.Subscriber()

    calib()
    
    of = obj_file.ObjFile('data/spray.obj')
    mesh = of.read()

    vertices = list(map(list, mesh.vertices))
    normals = list(map(list, mesh.normals))
    candidate_grasps = find_candidates(vertices, normals)
    
    inds = np.random.randint(0, len(candidate_grasps), 5)
    
    ###### Test Harness ######
    g = candidate_grasp[inds[0]]
    t, q = contacts_to_baxter_hand_pose(g[:, 0], g[:, 1])
    # t and q are the gripper position relative to bottle so convert that 
    # to the base frame by going through the ar_tag and then ar_tag location
    # and then tell moveit to go to that location
    bottle_loc = None
    while bottle_loc == None:
        try:
            bottle_loc = listener.lookupTransform('graspable_object', 'ar_marker_8', rospy.Time(0)eou
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue
    # Richard needs to figure out the math and do all the object conversions
    # Figure out how to open and close grippers and then just record
 
