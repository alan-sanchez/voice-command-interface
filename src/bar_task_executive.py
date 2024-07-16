#!/usr/bin/env python

## Import modules
import rospy
import sys
import os
import signal
import tf
import json
import actionlib
import moveit_commander 
import moveit_msgs.msg

## Import message types and other python libraries
from moveit_python import PlanningSceneInterface, MoveGroupInterface
from moveit_msgs.msg import MoveItErrorCodes
from geometry_msgs.msg import  Pose, Point, Quaternion
from robot_controllers_msgs.msg import QueryControllerStatesAction, QueryControllerStatesGoal, ControllerState
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


# Define a class attribute or a global variable as a flag
interrupted = False

# Define a signal handler function
def signal_handler(signal, frame):
    global interrupted
    interrupted = True

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)

class MoveGroupClient:
    '''
    Class for controlling mid-level operations of the arm.
    '''
    def __init__(self):
        '''
        A function that ...
        :param self: The self reference.
        '''
        rospy.loginfo("Waiting for MoveIt!")
        self.client = MoveGroupInterface("arm_with_torso", "base_link")

        ## First initialize `moveit_commander`
        moveit_commander.roscpp_initialize(sys.argv)

        ## Initialize publisher
        self.waypoint_sub = rospy.Subscriber('waypoint', Pose, self.callback)

        ## Instantiate a `RobotCommander`_ object. This object is the outer-level
        ## interface to the robot
        self.robot = moveit_commander.RobotCommander()

        ## Instantiate a `MoveGroupCommander`_ object.  This object is an interface
        ## to one group of joints.
        self.group = moveit_commander.MoveGroupCommander("arm_with_torso")
        self.group.set_end_effector_link("gripper_link") #use gripper_link if it is planar disinfection

        ## We create a `DisplayTrajectory`_ publisher which is used later to publish
        ## trajectories for RViz to visualize:
        self.display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                                       moveit_msgs.msg.DisplayTrajectory,
                                                       queue_size=20)
        rospy.loginfo("...connected")

        ## Padding does not work (especially for self collisions)
        ## So we are adding a box above the base of the robot
        self.scene = PlanningSceneInterface("base_link")
        self.scene.addBox("keepout", 0.25, 0.5, 0.09, 0.15, 0.0, 0.375)

        ## Allow replanning to increase the odds of a solution
        self.group.allow_replanning(True)

        ## Set the maximum velocity and acceleration scaling factors
        self.group.set_max_velocity_scaling_factor(0.1)  # 10% of the maximum speed
        self.group.set_max_acceleration_scaling_factor(0.5)  # 10% of the maximum acceleration

        ## Create the list of joints
        self.joints = ["torso_lift_joint", "shoulder_pan_joint", "shoulder_lift_joint", "upperarm_roll_joint",
                  "elbow_flex_joint", "forearm_roll_joint", "wrist_flex_joint", "wrist_roll_joint"]
                
    def callback(self, pose):
        # raw_input("Press enter to move pose")
        state = self.move_to_pose(pose)
        rospy.sleep(2)
        self.init_pose()

    def init_pose(self, vel = 0.2):
        """
        Function that sends a joint goal that moves the Fetch's arm and torso to
        the initial position.
        :param self: The self reference.
        :param vel: Float value for arm velocity.
        """
        ## List of joint values of init position
        pose =[.27, 1.41, 0.30, -0.22, -2.25, -1.56, 1.80, -0.37,]
        while not rospy.is_shutdown():
            result = self.client.moveToJointPosition(self.joints,
                                                     pose,
                                                     0.0,
                                                     wait=True,
                                                     max_velocity_scaling_factor=vel)
            if result and result.error_code.val == MoveItErrorCodes.SUCCESS:
                self.scene.removeCollisionObject("keepout")
                return 0 

    def move_to_pose(self, pose):
        # Set the target pose
        self.group.set_pose_target(pose)

        # Plan the motion to the target pose
        plan = self.group.plan()

        # print(plan)
        # Execute the planned motion
        success = self.group.go(wait=True)

        # # Stop any residual movement
        # self.group.stop()

        # # Clear the pose target
        # self.group.clear_pose_targets()

        return success

    
class PointHeadClient(object):
    def __init__(self):
        self.client = actionlib.SimpleActionClient("head_controller/follow_joint_trajectory", FollowJointTrajectoryAction)
        rospy.loginfo("Waiting for head_controller...")
        self.client.wait_for_server()

    def set_pan_tilt(self, pan=-0.02, tilt=0.86, duration=1.0):
        # Create a trajectory point
        point = JointTrajectoryPoint()
        point.positions = [pan, tilt]
        point.time_from_start = rospy.Duration(duration)

        # Create a trajectory
        trajectory = JointTrajectory()
        trajectory.joint_names = ["head_pan_joint", "head_tilt_joint"]
        trajectory.points = [point]

        # Create a goal
        goal = FollowJointTrajectoryGoal()
        goal.trajectory = trajectory

        # Send the goal to the action server
        self.client.send_goal(goal)
        self.client.wait_for_result()



if __name__ == '__main__':
    ## Initialize the `relax_arm_control` node
    rospy.init_node('bar_task_executive')

    ## Instantiate the `ArmControl()` object
    body_obj = MoveGroupClient()
    head_obj = PointHeadClient()
    body_obj.init_pose()
    head_obj.set_pan_tilt()

    rospy.spin()
    # raw_input("Press enter to move Fetch to it's initial arm configuration")
    # obj.init_pose()
    # print("")
    
    # obj.relax_arm()
    # raw_input("My arm is in relax mode. You can move it and hover it above the center top of the object you want me to disinfect. Once you have done that, press Enter.")
    
    # raw_input("I will start recording the cleaning task once you press enter")
    # obj.record()
    ## Notify user that they can move the arm
    # rospy.loginfo("Relaxed arm node activated. You can now move the manipulator")
    # print()

    # obj.playback()
    # rospy.loginfo("Type Ctrl + C when you are done recording")
    # rospy.spin()