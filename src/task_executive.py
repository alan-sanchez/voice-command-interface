#!/usr/bin/env python

## Import modules
import rospy
import sys
import os
import signal
import tf
import json
import ast
import actionlib
import moveit_commander 
import moveit_msgs.msg

## Import message types and other python libraries
from moveit_python import MoveGroupInterface, PlanningSceneInterface
from moveit_msgs.msg import MoveItErrorCodes
from geometry_msgs.msg import  Pose, Point, Quaternion, PoseStamped
from std_msgs.msg import String, Header
from robot_controllers_msgs.msg import QueryControllerStatesAction, QueryControllerStatesGoal, ControllerState
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
  

class MoveGroupClient:
    '''
    Class for controlling mid-level operations of the arm and torso.
    '''
    def __init__(self):
        '''
        A function that initializes the MoveGroupClient.
        '''
        ## Notify user for Moveit connection
        rospy.loginfo("Waiting for MoveIt!")

        self.move_group = MoveGroupInterface("arm_with_torso", "base_link")
        self.gripper_frame = ("gripper_link")

        # ## Padding does not work (especially for self collisions), so we are adding a box above the base of the robot
        self.planning_scene = PlanningSceneInterface("base_link")
        self.planning_scene.addBox("keepout", 0.25, 0.5, 0.09, 0.15, 0.0, 0.375)
        self.planning_scene.addBox("table", 0.6, 1.5, 0.05, 0.69, 0.0, 0.75)

      
    def init_pose(self, vel=0.2):
        """
        Function that sends a joint goal that moves the Fetch's arm and torso to the initial position.

        Parameters:
        - self: The self reference.
        - vel (float): The arm velocity scaling factor.
        """
        ## List of joint names
        joints = ["torso_lift_joint", "shoulder_pan_joint", "shoulder_lift_joint", "upperarm_roll_joint",
                  "elbow_flex_joint", "forearm_roll_joint", "wrist_flex_joint",    "wrist_roll_joint"]

        ## List of joint values for initial position
        pose = [.27, 1.41, 0.30, -0.22, -2.25, -1.56, 1.80, -0.37]

        while not rospy.is_shutdown():
            ## move to the inital joint position
            result = self.move_group.moveToJointPosition(joints,
                                                         pose,
                                                         0.0,
                                                         wait=True,
                                                         max_velocity_scaling_factor=vel)
            if result and result.error_code.val == MoveItErrorCodes.SUCCESS:
                ## Remove collision object after reaching the initial position
                self.planning_scene.removeCollisionObject("keepout")
                return 0 
 


class PointHeadClient:
    '''
    Class for controlling the head movements.
    '''
    def __init__(self):
        '''
        Initialize the PoinHeadClient.
        '''
        ## Create an action client to send goals the the head controller
        self.head_client = actionlib.SimpleActionClient("head_controller/follow_joint_trajectory", FollowJointTrajectoryAction)
        rospy.loginfo("Waiting for head_controller...")
        self.head_client.wait_for_server()
        rospy.loginfo("...connected")


    def set_pan_tilt(self, pan=0.00, tilt=0.86, duration=1.0):
        '''
        Set the pan and tilt for the head.

        Parameters:
        -self: The self reference.
        -pan (float): The pan angle.
        -tilt (float): The tilt angle.
        -duration (float): The duration for the movement.
        '''
        ## Create a trajectory point
        point = JointTrajectoryPoint()
        point.positions = [pan, tilt]
        point.time_from_start = rospy.Duration(duration)

        ## Create a trajectory
        trajectory = JointTrajectory()
        trajectory.joint_names = ["head_pan_joint", "head_tilt_joint"]
        trajectory.points = [point]

        ## Create a goal
        goal = FollowJointTrajectoryGoal()
        goal.trajectory = trajectory

        ## Send the goal to the action server
        self.head_client.send_goal(goal)
        self.head_client.wait_for_result()



class TaskExecutive(MoveGroupClient, PointHeadClient):
    '''
    Class for task execution combining arm and head control.
    '''
    def __init__(self):
        ''' 
        Initialize the TaskExecutive.
        '''
        MoveGroupClient.__init__(self)
        PointHeadClient.__init__(self)

        ## Command robot to start at initial arm, torso, and head position
        self.init_pose()
        self.set_pan_tilt()

        ## Initialize subscibers
        self.known_obj_sub     = rospy.Subscriber('known_object_dict', String, self.waypoint_generator)
        self.init_pose_cmd_sub = rospy.Subscriber('human_demo_status', String, self.init_pose_callback)
        
        ## Initial publisher
        self.cleaning_status_pub = rospy.Publisher('cleaning_status', String, queue_size=10)
       
        # Setup header
        self.header = Header()
        self.header.frame_id = "/base_link"
        self.header.stamp = rospy.Time.now()
    
        ## Initialize dictionary
        self.contaminated_obj_dict = {}

        ## Initialize table height where the objects were tested on
        self.test_table_height = 0.77

        ## Specify the relative and audio directory paths
        self.relative_path = 'catkin_ws/src/voice_command_interface/tool_paths'        

        ## Log initialization notifier
        rospy.loginfo('{}: is ready.'.format(self.__class__.__name__))


    def init_pose_callback(self, str_msg):
        '''
        Callback function to reset the robot to the initial pose.

        Parameters:
        - self: The self reference.
        - str_msg (String): Message containing status of human demo.
        '''
        if str_msg.data == 'init_pose':
            attribute = self.init_pose()
            print(attribute)


    def waypoint_generator(self,msg):
        '''
        Generate waypoints for the robot to follow based on the contaminated object dictionary.

        Parameters:
        - self: The self reference.
        - msg(String): A String containing the information of the contaminated objects.
        '''
        ## Convert the message data to a dictionary
        contaminated_obj_dict = ast.literal_eval(msg.data)
        print(contaminated_obj_dict)
        
        ## Create empty lists for waypoints
        waypoints = []

        ## Search for tool paths for each object
        for key, value in contaminated_obj_dict.items():
            file_path = os.path.join(os.environ['HOME'],self.relative_path, key + '.json')

            if os.path.exists(file_path):
                with open(file_path) as user_file:
                    poses = json.load(user_file)
                    # print(len(poses))

                ## Calculate shifts required to align the pre-recorded path with the current object position
                shift_x = round(value['centroid'][0] - poses[0][0], 2)
                shift_y = round(value['centroid'][1] - poses[0][1], 2)
                shift_z = round(value['table_height'] - self.test_table_height, 2)

                ## Append each new pose to the list of waypoints
                for i, pose in enumerate(poses):
                    new_pose = Pose(Point(pose[0] + shift_x, pose[1] + shift_y, pose[2] + shift_z),
                                          Quaternion(pose[3], pose[4], pose[5], pose[6])
                                    )
                    waypoints.append(new_pose)

        ## Initiate `move_along_waypoints` method
        self.move_along_waypoints(waypoints)


    def move_along_waypoints(self, waypoints):
        '''
        Move the rbot along the specified waypoints.

        Parameters:
        - self: The self reference:
        - waypoints (List): List containing Pose message types. 
        '''
        ## Construct a "pose_stamped" message as required by moveToPose
        gripper_pose_stamped = PoseStamped()
        gripper_pose_stamped.header.frame_id = "base_link"

        ## Publish cleaning status so the robot doesn't update map while cleaning
        self.cleaning_status_pub.publish('cleaning')
        
        for waypoint in waypoints:    
            ## Update header to current time stamp
            gripper_pose_stamped.header.stamp = rospy.Time.now()

            ## Set the gripper pose
            gripper_pose_stamped.pose = waypoint

            ## Move to the pose
            self.move_group.moveToPose(pose_stamped=gripper_pose_stamped,
                                       gripper_frame=self.gripper_frame,
                                       tolerance=0.1,
                                       max_velocity_scaling_factor=0.2,)

            result = self.move_group.get_move_action().get_result()

            if result:
                ## Checking the MoveItErrorCode
                if result.error_code.val == MoveItErrorCodes.SUCCESS:
                    # rospy.loginfo("Made it to waypoint")
                    flag = True
                else:
                    ## If you get to this point please search for: moveit_msgs/MoveItErrorCodes.msg
                    rospy.logerr("Arm goal in state: %s",
                                 self.move_group.get_move_action().get_state())
            else:
                rospy.logerr("MoveIt! failure no result returned.")
        
        ## Reset to initial pose and publish completion status
        self.init_pose()
        self.cleaning_status_pub.publish('complete')
        


if __name__ == '__main__':
    ## Initialize the `relax_arm_control` node
    rospy.init_node('task_2_executive')

    ## Instantiate the `TaskExecutive` object
    obj = TaskExecutive()
    
    ## Keep the node running
    rospy.spin()
