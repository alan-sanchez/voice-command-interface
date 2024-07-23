#!/usr/bin/env python

## Import modules
import rospy
import os
import json
import ast
import actionlib


## Import message types and other python libraries
from moveit_python import MoveGroupInterface, PlanningSceneInterface
from moveit_msgs.msg import MoveItErrorCodes
from geometry_msgs.msg import  Pose, Point, Quaternion, PoseStamped
from std_msgs.msg import String, Header
from visualization_msgs.msg import Marker, MarkerArray
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
        """
        ##
        joints = ["torso_lift_joint", "shoulder_pan_joint", "shoulder_lift_joint", "upperarm_roll_joint",
                  "elbow_flex_joint", "forearm_roll_joint", "wrist_flex_joint",    "wrist_roll_joint"]

        ## List of joint values for initial position
        pose = [.27, 1.41, 0.30, -0.22, -2.25, -1.56, 1.80, -0.37]

        while not rospy.is_shutdown():
            result = self.move_group.moveToJointPosition(joints,
                                                         pose,
                                                         0.0,
                                                         max_velocity_scaling_factor=vel)
            if result and result.error_code.val == MoveItErrorCodes.SUCCESS:
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
        self.head_client = actionlib.SimpleActionClient("head_controller/follow_joint_trajectory", FollowJointTrajectoryAction)
        rospy.loginfo("Waiting for head_controller...")
        self.head_client.wait_for_server()
        rospy.loginfo("...connected")

    def set_pan_tilt(self, pan=0.00, tilt=0.86, duration=1.0):
        '''
        Set the pan and tilt for the head.

        Parameters:
        pan ():
        tilte ():
        duration()

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
        
        '''
        MoveGroupClient.__init__(self)
        PointHeadClient.__init__(self)

        #
        self.init_pose()
        self.set_pan_tilt()

        ## Initialize subsciber
        self.known_obj_sub = rospy.Subscriber('known_object_dict',String, self.waypoint_generator)

        self.cleaning_status_pub = rospy.Publisher('cleaning_status', String, queue_size=10)

        ##
        # self.waypoints_marker_pub = rospy.Publisher('waypoints_marker', Marker    , queue_size=1)
        self.waypoints_marker_pub = rospy.Publisher('waypoints', MarkerArray, queue_size=1)
        # Setup header
        self.header = Header()
        self.header.frame_id = "/base_link"
        self.header.stamp = rospy.Time.now()
    
        # Initialize waypoint_markers and all of the other feature values
        self.waypoints_marker = Marker()
        self.waypoints_marker.header = self.header
        self.waypoints_marker.type = Marker.ARROW
        self.waypoints_marker.scale.x = 0.03
        self.waypoints_marker.scale.y = 0.01
        self.waypoints_marker.scale.z = 0.005
        self.waypoints_marker.color.a = 1
        self.waypoints_marker.color.r = 0
        self.waypoints_marker.color.g = 0
        self.waypoints_marker.color.b = 1.0

        ## Initialize dictionary
        self.contaminated_obj_dict = {}

        ## Initialize table height where the objects were tested on
        self.test_table_height = 0.77

        ## Specify the relative and audio directory paths
        self.relative_path = 'catkin_ws/src/voice_command_interface/tool_paths'        

        ## Log initialization notifier
        rospy.loginfo('{}: is ready.'.format(self.__class__.__name__))



    def waypoint_generator(self,msg):
        '''

        '''
        ##
        contaminated_obj_dict = ast.literal_eval(msg.data)

        waypoints = []
        marker_array = MarkerArray()

        for key, value in contaminated_obj_dict.items():
            file_path = os.path.join(os.environ['HOME'],self.relative_path, key + '.json')

            if os.path.exists(file_path):
                with open(file_path) as user_file:
                    poses = json.load(user_file)
                    # print(len(poses))

                # Calculate shifts required to align the pre-recorded path with the current object position
                shift_x = round(value['centroid'][0] - poses[0][0], 2)
                shift_y = round(value['centroid'][1] - poses[0][1], 2)
                shift_z = round(value['table_height'] - self.test_table_height, 2)

                for i, pose in enumerate(poses):
                    new_pose = Pose(Point(pose[0] + shift_x, pose[1] + shift_y, pose[2] + shift_z),
                                          Quaternion(pose[3], pose[4], pose[5], pose[6])
                                    )
                    waypoints.append(new_pose)

                    # Create and append a marker for visualization
                    marker = Marker()
                    marker.header.frame_id = "base_link"
                    marker.header.stamp = rospy.Time.now()
                    marker.ns = "waypoints"
                    marker.id = i
                    marker.type = Marker.SPHERE
                    marker.action = Marker.ADD
                    marker.pose = new_pose
                    marker.scale.x = 0.01
                    marker.scale.y = 0.01
                    marker.scale.z = 0.01
                    marker.color.a = 1.0
                    marker.color.r = 1.0
                    marker.color.g = 0.0
                    marker.color.b = 0.0
                    marker_array.markers.append(marker)


        self.waypoints_marker_pub.publish(marker_array)
        # print(shift_z)
        # rospy.loginfo("Published waypoints markers")
        self.move_along_waypoints(waypoints)


    def move_along_waypoints(self, waypoints):

        gripper_pose_stamped = PoseStamped()
        gripper_pose_stamped.header.frame_id = "base_link"

        self.cleaning_status_pub.publish('cleaning')
        for waypoint in waypoints:    
            ##
            gripper_pose_stamped.header.stamp = rospy.Time.now()

            ## 
            gripper_pose_stamped.pose = waypoint

            self.move_group.moveToPose(pose_stamped=gripper_pose_stamped,
                                       gripper_frame=self.gripper_frame,
                                       tolerance=0.1,
                                       max_velocity_scaling_factor=0.2,)

            result = self.move_group.get_move_action().get_result()

            if result:
                # Checking the MoveItErrorCode
                if result.error_code.val == MoveItErrorCodes.SUCCESS:
                    # rospy.loginfo("Made it to waypoint")
                    flag = True
                else:
                    # If you get to this point please search for:
                    # moveit_msgs/MoveItErrorCodes.msg
                    rospy.logerr("Arm goal in state: %s",
                                 self.move_group.get_move_action().get_state())
            else:
                rospy.logerr("MoveIt! failure no result returned.")

        # This stops all arm movement goals
        # It should be called when a program is exiting so movement stops
        # self.move_group.get_move_action().cancel_all_goals()

        
        self.init_pose()
        self.cleaning_status_pub.publish('complete')
        


if __name__ == '__main__':
    ## Initialize the `relax_arm_control` node
    rospy.init_node('task_1_executive')

    obj = TaskExecutive()
    

    rospy.spin()
