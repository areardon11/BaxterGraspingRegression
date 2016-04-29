#!/usr/bin/env python
import sys
import rospy
import moveit_commander
import tf
from tf.transformations import *
import obj_file
import numpy as np
from geometry_msgs.msg import PoseStamped
from baxter_interface import gripper as baxter_gripper
import baxter_interface
import pickle 
from moveit_msgs.msg import *
from fc import *

# the name of the world frame
BASE_FRAME = 'base'
HEIGHT_OFFSET = 1.0
OPEN_AMOUNT = 100.0
hld_frc = 1
mv_frc = 25

def get_pose(arm):
    arm = left_arm
    pose = arm.endpoint_pose()
    pos = pose['position']
    Q = pose['orientation']
    return pos.x, pos.y, pos.z, [Q.x, Q.y, Q.z, Q.w]

def lookup_transform(name):
    while True:
        try:
            trans, rot = listener.lookupTransform(BASE_FRAME, name, rospy.Time(0))
            return list(trans), list(rot)
        except:
            continue
    #x, y, z, rott = get_pose(name)
    #trans = np.array(trans) + np.array([x,y,z])
    #rot = np.array(rot) + np.array(rott)
    #return trans/2, rot/2

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

def check_grasp():
    return 1

def calib():
    global pregrasp_pose
    pregrasp_pose = None

    # f = open('def_pre_pose','rb')
    # pregrasp_pose = pickle.load(f)
    # f.close()
    # print("Default Imaging Pose = ", pregrasp_pose)
    
    # goto_pregrasp_pose()
    raw_input('Press <Enter> to save location of PreGrasp Pose.')
    with open('def_pre_pose', 'w+') as f:
        while pregrasp_pose == None:
                try:
                    pregrasp_pose = lookup_transform('left_hand')
                except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                    continue
        print("Prepose = ", pregrasp_pose)
        pickle.dump(pregrasp_pose, f)
    
    captured = False
    raw_input('Press <Enter> to save location of PrePose 1')
    with open('prepose_1', 'w+') as f:
        while not captured:
            try:
                pose = lookup_transform('left_hand')
                captured = True
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                print("What is going on??")
                continue
        pickle.dump(pose, f)
    
    return pregrasp_pose, pose

if __name__ == "__main__":
    moveit_commander.roscpp_initialize(sys.argv)
    
    rospy.init_node("grasping")
    
    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()
    left_arm = baxter_interface.Limb('left')
    left_planner = moveit_commander.MoveGroupCommander('left_arm')
    left_planner.set_planner_id('RRTConnectkConfigDefault')
    left_planner.set_planning_time(10)
    
    left_gripper = baxter_gripper.Gripper('left')
#   left_gripper.calibrate()
    #left_gripper.set_holding_force(hld_frc)
    #left_gripper.set_moving_force(mv_frc)

    listener = tf.TransformListener()
    br = tf.TransformBroadcaster()
    
    #rospy.Subscriber()

    #pre_grasp_pose, pre_one = calib()
    
    of = obj_file.ObjFile('data/spray.obj')
    mesh = of.read()

    vertices = list(map(list, mesh.vertices))
    normals = list(map(list, mesh.normals))
    candidate_grasps = find_candidates(vertices, normals)
    
    
    ###### Test Harness ######
    for i in range(10):
        raw_input("Run test: " + str(i))        
        g = candidate_grasps[4]
        t, q = contacts_to_baxter_hand_pose(g[0], g[1])
        R0 = quaternion_matrix(q)
        T0 = translation_matrix(t)
        # t and q are the gripper position relative to bottle so convert that 
        # to the base frame by going through the ar_tag and then ar_tag location
        # and then tell moveit to go to that location (.504, .544, -.471, .478)
        moved = False
        #q[0] = -q[0]
        #q[1] = -q[1]
        pre_t = t - np.array((.2,0,0))
        t -= np.array((.05,0,0))
        lift_t = t + np.array((0,0,.2))
        pre_trans, pre_rot = None, None
        while not moved:
            try:
                br.sendTransform(pre_t, q, rospy.Time.now(), "pre_grasp", "graspable_object")
                pre_trans, pre_rot = listener.lookupTransform('base', 'pre_grasp', rospy.Time(0))
                goto(pre_trans, pre_rot)
                moved = True
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                print(e)
                continue
        
        release()
        
        moved = False
        trans, rot = None, None
        while not moved:
            try:
                br.sendTransform(t, q, rospy.Time.now(), "grasp", "graspable_object")
                trans, rot = listener.lookupTransform('base', 'grasp', rospy.Time(0))
                goto(trans, rot)
                moved = True
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                print(e)
                continue

        grasp()
        goto((trans[0], trans[1], trans[2] + .2), rot)

        rospy.sleep(.5)

        goto(trans, rot)
        release()
        goto(pre_trans, pre_rot)

    #check_grasp()

    #raw_input("going to pregrasp")
    #goto_pregrasp_pose()
    #raw_input("Going to next pose")
    #goto(pre_one[0], pre_one[1])
    
