#!/usr/bin/env python
import roslib
import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import pickle
import tf
from tf.transformations import *
import numpy as np
from std_msgs import msg
from IPython import embed
from pc_generator import pc_gener
from point_cloud_sampler import *
import moveit_commander
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion, TransformStamped
from baxter_interface import gripper as baxter_gripper
import baxter_interface
from moveit_msgs.msg import *

BASE_FRAME = 'base'
HEIGHT_OFFSET = np.array([0.0, 0.0, 0.3])
PREGRASP_OFFSET = np.array([-0.1, 0.0, 0.0])
CAM_GRIP_OFFSET = np.array([-0.0762,0.0,0.0])

def lookup_transform(name):
    while True:
        try:
            trans, rot = listener.lookupTransform(BASE_FRAME, name, rospy.Time(0))
            break
        except:
            continue
    return [trans, rot]

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
    rot = np.array(rot)
    trans = np.array(trans)
    planner = left_planner

    #goal = PoseStamped()
    #goal.header.frame_id = BASE_FRAME

    # assign_xyz(trans, goal.pose.position)
    # assign_xyz(rot, goal.pose.orientation)
    goal = Pose()
    goal.position.x = trans[0]
    goal.position.y = trans[1]
    goal.position.z = trans[2]
    goal.orientation.x = rot[0]
    goal.orientation.y = rot[1]
    goal.orientation.z = rot[2]
    goal.orientation.w = rot[3]

    # find a plan to get there
    planner.clear_pose_targets()
    planner.set_pose_target(goal)
    planner.set_start_state_to_current_state()
    plan = planner.plan()
    planner.execute(plan)
    rospy.sleep(1.5)
    
def lift():
    lift_t, lift_r = lookup_transform('left_hand')
    lift_t = lift_t + HEIGHT_OFFSET
    goto(lift_t, lift_r)

def grasp():
    left_gripper.close(block=True)
    rospy.sleep(.5)

def release():
    left_gripper.open(block=True)
    #left_gripper.command_position(OPEN_AMOUNT, block=True)
    rospy.sleep(.5)

def execute_grasp(trans, rot):
    pregrasp_t = trans + PREGRASP_OFFSET
    pregrasp_r = rot

    print 'Goal Trans:', trans
    print 'Goal Rot:', rot
    #print 'Pregrasp Trans:', pregrasp_t
    #print 'Pregrasp Rot:', pregrasp_r

    goto(pregrasp_t, pregrasp_r)

    goto(trans, rot)

    grasp()
    lift()
    raw_input('Press Enter to put object down.')
    new_t = trans + np.array([0.0,0.0,0.025])
    goto(new_t, rot)
    release()
    new_t2 = new_t + PREGRASP_OFFSET
    goto(new_t2, rot)
    print('Done executing grasp.')

def visualize_pc(pc):
    pcl_pub = rospy.Publisher("/custom_pc", PointCloud2, queue_size=10)
    points = pc.tolist()
    rate = rospy.Rate(1)
    while not rospy.is_shutdown():
        header = msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'base' 
        point_cloud = pc2.create_cloud_xyz32(header, points)
        pcl_pub.publish(point_cloud)
        rate.sleep()

def test():
    pc_gen.gen_new_pc()
    pc = pc_gen.get_pc()
    t, r = determine_grasp(pc)
    # t = t + CAM_GRIP_OFFSET
    t0,r0 = lookup_transform('left_hand')
    execute_grasp(t,r0)
    goto(start_t, start_r)

if __name__ == '__main__':
    print('Here0')
    rospy.init_node('master', anonymous=True)
    moveit_commander.roscpp_initialize(sys.argv)
    robot = moveit_commander.RobotCommander()
    #scene = moveit_commander.PlanningSceneInterface()
    #left_arm = baxter_interface.Limb('left')
    left_planner = moveit_commander.MoveGroupCommander('left_arm')
    left_planner.set_planner_id('RRTConnectkConfigDefault')
    left_planner.set_planning_time(10)
    left_gripper = baxter_gripper.Gripper('left')
    listener = tf.TransformListener()
    print('Here1')
    grasp()
    release()
    start_t, start_r = lookup_transform('left_hand')
    pc_gen = pc_gener()
    default_t = (0.29538318984501866, 0.5142008508685174, -0.07371889323629975)
    default_r = (0.01736513852756305, 0.8090332012510187, 0.026711162432059874, 0.5868988371422927)
    goto(default_t, default_r)
    while not rospy.is_shutdown():
        raw_input('Press <Enter> to Plan a new grasp...')
        test()