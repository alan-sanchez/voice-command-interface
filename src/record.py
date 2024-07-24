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

## Define a class attribute or a global variable as a flag
interrupted = False

## Function to handle signal interruption
def signal_handler(signal, frame):
    global interrupted
    interrupted = True

## Assign the signal handler to the SIGINT signal
signal.signal(signal.SIGINT, signal_handler)

class Record:
    '''
    Class for controlling mid-level operations of the arm, including moving to initial positions,
    recording movements, and playing back movements.
    '''
    def __init__(self):
        '''
        Initializes the robot arm control, setting up MoveIt interfaces and other necessary components.

        Parameters:
        - self: The self reference.
        '''
        rospy.loginfo("Waiting for MoveIt!")
        self.client = MoveGroupInterface("arm_with_torso", "base_link")

        ## Initialize `moveit_commander`
        moveit_commander.roscpp_initialize(sys.argv)

        ## Instantiate a `RobotCommander`_ object, the outer-level interface to the robot
        self.robot = moveit_commander.RobotCommander()

        ## Instantiate a `MoveGroupCommander`_ object, an interface to group of joints
        self.group = moveit_commander.MoveGroupCommander("arm_with_torso")
        self.group.set_end_effector_link("gripper_link") #use gripper_link if it is planar disinfection

        ## Create a `DisplayTrajectory`_ publisher to publish trajectories for RViz to visualize
        self.display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                                       moveit_msgs.msg.DisplayTrajectory,
                                                       queue_size=20)
        rospy.loginfo("...connected")

        ## Setup enviroment around the robot and add a box to the scene
        self.scene = PlanningSceneInterface("base_link")
        self.scene.addBox("keepout", 0.25, 0.5, 0.09, 0.15, 0.0, 0.375)

        ## Create the list of joints
        self.joints = ["torso_lift_joint", "shoulder_pan_joint", "shoulder_lift_joint", "upperarm_roll_joint",
                  "elbow_flex_joint", "forearm_roll_joint", "wrist_flex_joint", "wrist_roll_joint"]
        
        ## Instantiate a `TranformListener` object
        self.listener = tf.TransformListener()

        # Contrct the path to save recorded movements
        relative_path = 'catkin_ws/src/voice_command_interface/tool_paths'
        self.full_path = os.path.join(os.environ['HOME'], relative_path)
        self.file = None
        
        
        ## Action client for querying controller states
        controller_states = "/query_controller_states"
        self._controller_client = actionlib.SimpleActionClient(controller_states, QueryControllerStatesAction)
        self._controller_client.wait_for_server()

        ## Define controllers for gravity compensation
        self._gravity_comp_controllers = ["arm_controller/gravity_compensation"]
        self._non_gravity_comp_controllers = list()
        self._non_gravity_comp_controllers.append("arm_controller/follow_joint_trajectory")
        self._non_gravity_comp_controllers.append("arm_with_torso_controller/follow_joint_trajectory")


    def init_pose(self, vel = 0.2):
        """
        Function that sends a joint goal that moves the Fetch's arm and torso to
        the initial position.

        Parameters:
        - self: The self reference.
        - vel (Float): Arm velocity scaling factor.
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


    def relax_arm(self):
        '''
        Turns on gravity compensation controller and turns
        off other controllers

        Parameters:
        - self: The self reference.
        '''
        ## Initialize a goal for the QueryControllerStates action,
        goal = QueryControllerStatesGoal()

        ## Loop over the list of controllers that should be in gravity compensation mode
        for controller in self._gravity_comp_controllers:
            state = ControllerState()
            state.name = controller
            state.state = state.RUNNING
            goal.updates.append(state)

        ## Loop over the list of controllers that should not be in gravity compensation mode
        for controller in self._non_gravity_comp_controllers:
            state = ControllerState()
            state.name = controller
            state.state = state.STOPPED
            goal.updates.append(state)
        
        ## Send the assembled goal to the action server responsible for controller state management
        self._controller_client.send_goal(goal)


    def record(self):
        '''
        A function that records the human-guided motions

        Parameters:
        - self: The self reference.
        '''
        global interrupted
        pose_arr = []
        print("\nRecording movement...\n")
        print("\nPress ctrl+c to stop recording and save to file\n")
        while not interrupted:
            joints_arr = self.ee_pose()
            # Check if all joint values are not zero before appending
            if any(value != 0 for value in joints_arr):
                pose_arr.append(joints_arr)
            rospy.sleep(1)
        
        file_name = raw_input('\nname your movement file: ')

        ## Construct the full file path
        file_path = os.path.join(self.full_path, file_name + '.json')

        ## 
        json_object = json.dumps(pose_arr, indent=4)
        with open(file_path, 'w') as outfile:
            outfile.write(json_object)
        
        ## Reset interrupted flag
        interrupted = False


    def ee_pose(self):
        '''
        Function thatfinds the pose of the `gripper_link` relative to the `base_link` frame.
        :param self: The self reference.

        :return [trans, rot]: The pose message type.
        '''
        while not rospy.is_shutdown():
            try:
                (trans,rot) = self.listener.lookupTransform( '/base_link', '/gripper_link',rospy.Time(0))
                trans = [round(p,3) for p in trans]
                rot = [round(r,2) for r in rot]
                return trans + rot

            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                pass


    def playback(self):
        file_name = raw_input('\nEnter the name of the movement file you wish to replay: ')
        ## Construct the full file path
        file_path = os.path.join(self.full_path, file_name + '.json')
        print(file_path,type(file_path))
        with open(file_path) as user_file:
            poses = json.load(user_file)
        
        # print(poses_object)
        # print(len(poses),type(poses))
        waypoints = []
        for ps in poses:
            new_pose = Pose(Point(ps[0], ps[1], ps[2]),Quaternion(ps[3], ps[4], ps[5], ps[6]))
            waypoints.append(new_pose)
        
        # print(waypoints, type(waypoints))
        # # allow replanning until the waypoints reached in plan is over 90%
        fraction = 0
        count = 0
        
        (plan, fraction) = self.group.compute_cartesian_path(waypoints, # waypoints to follow
                                                        0.01,      # eef_step
                                                        0.00)      # jump_threshold

        plan = self.group.retime_trajectory(self.robot.get_current_state(),
                                        plan,
                                        velocity_scaling_factor = 0.2,
                                        acceleration_scaling_factor = 0.2,
                                        )

        if fraction > 0.9:
            self.group.execute(plan, wait=True)
        else:
            rospy.WARN("Could not plan the cartesian path")


if __name__ == '__main__':
    ## Initialize the `relax_arm_control` node
    rospy.init_node('record')

    ## Instantiate the `ArmControl()` object
    obj = Record()

    # outer loop controlling create movement or play movement
    while(True):
        print("\n\nEnter 1 to move to the initial arm configuration\nEnter 2 to record a motion\nEnter 3 to playback a trajectory\nEnter 4 to quit\n")
        control_selection = input("Choose an option: ")

        ## sub loop controlling add a movement feature
        if control_selection == 1:
            obj.init_pose()
                        
        ## play back a movement from file
        elif control_selection == 2:
            print("My arm is in relax mode. \nHover my hand 6 inches above the object you want me to disinfect.")
            obj.relax_arm()

            raw_input("I will start recording the cleaning task once you press enter")
            obj.record()

        elif control_selection == 3:
            obj.playback()

        elif control_selection == 4:
            break

        else:
            print("\nInvalid selection\n")
    

    # # raw_input("Press enter to move Fetch to it's initial arm configuration")
    # # obj.init_pose()
    # # print("")
    
    # # raw_input("Press Enter to have my arm is in relax mode. \nHove my hand 6 inches above the object you want me to disinfect.")
    

    
    # rospy.loginfo("Type Ctrl + C when you are done recording")
    # rospy.spin()