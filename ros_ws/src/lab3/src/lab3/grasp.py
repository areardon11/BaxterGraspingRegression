#!/usr/bin/env python
import sys
import rospy
import moveit_commander
from geometry_msgs.msg import PoseStamped
from baxter_interface import gripper as baxter_gripper

#Initialize moveit_commander
moveit_commander.roscpp_initialize(sys.argv)
#Start a node
rospy.init_node('moveit_node')
#Initialize both arms
robot = moveit_commander.RobotCommander()
scene = moveit_commander.PlanningSceneInterface()
right_arm = moveit_commander.MoveGroupCommander('right_arm')
right_arm.set_planner_id('RRTConnectkConfigDefault')
right_arm.set_planning_time(10)

listener = tf.TransformListener()
ar_loc = None
#First goal pose ------------------------------------------------------
goal = PoseStamped()
#goal_pose.header.frame_id = "base"

#x, y, and z position
goal.pose.position.x = 0.2
goal.pose.position.y = 0.6
goal.pose.position.z = 0.2

#Orientation as a quaternion
goal.pose.orientation.x = 0.0
goal.pose.orientation.y = -1.0
goal.pose.orientation.z = 0.0
goal.pose.orientation.w = 0.0

#Set the goal state to the pose you just defined
right_arm.set_pose_target(goal)

#Set the start state for the left arm
right_arm.set_start_state_to_current_state()

#Plan a path
right_plan = right_arm.plan()

#Execute the plan
right_arm.execute(right_plan)

if __name__ == '__main__':
	# 1. Find position of AR tag relative to Baxter.
	# 2. Set Camera pose position.
	# 3. Get goal TF.
	# 4. Move to the goal TF and grasp object.
	# 5. Lift object and determine if grasp was successful.