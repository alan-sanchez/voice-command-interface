#!/usr/bin/env python

## Import modules
import rospy
import sys
import os
import ast
import tf
import json
import actionlib

## Import message types and other python libraries
from geometry_msgs.msg import  Pose, Point, Quaternion
from std_msgs.msg import String, Header
from robot_controllers_msgs.msg import QueryControllerStatesAction, QueryControllerStatesGoal, ControllerState


class ArmControl:
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
        ## Intialize subscriber
        self.unknown_object_sub    = rospy.Subscriber('unknown_object_label', String, self.label_callback)
        self.human_demo_status_sub = rospy.Subscriber('human_demo_status',    String, self.operations_callback)
       
        # Contrct the path to save recorded movements
        relative_path = 'catkin_ws/src/voice_command_interface/tool_paths'
        self.full_path = os.path.join(os.environ['HOME'], relative_path)
        self.file = None

        ## Instantiate a `TranformListener` object
        self.listener = tf.TransformListener()
            
        ## Action client for querying controller states
        controller_states = "/query_controller_states"
        self.controller_client = actionlib.SimpleActionClient(controller_states, QueryControllerStatesAction)
        self.controller_client.wait_for_server()

        ## Define controllers for gravity compensation
        self.gravity_comp_controllers = ["arm_controller/gravity_compensation"]
        self.non_gravity_comp_controllers = list()
        self.non_gravity_comp_controllers.append("arm_controller/follow_joint_trajectory")
        self.non_gravity_comp_controllers.append("arm_with_torso_controller/follow_joint_trajectory")

        ## Initialize a timer to periodically update locations
        self.timer = rospy.Timer(rospy.Duration(1), self.timer_callback)

        self.demo_status = None

        ## Log initialization notifier
        rospy.loginfo('{}: is ready.'.format(self.__class__.__name__))

    def label_callback(self, msg):
        '''
        
        '''
        # self.unknown_item_dict = json.loads(msg.data)#ast.literal_eval(msg.data)
        self.label = msg.data
       
    
    def operations_callback(self,msg):
        print(msg.data)
        if msg.data == 'relax':
            self.relax_arm()
        else:
            self.demo_status = msg.data

    
    def timer_callback(self, event):
        pose_list = []
        while self.demo_status == 'start':
            pose = self.ee_pose()
            if any(value != 0 for value in pose):
                pose_list.append(pose)
            rospy.sleep(1)

        if len(pose_list) > 1:
            ## Construct the full file path
            file_path = os.path.join(self.full_path, str(self.label) + '.json')

            ## 
            json_object = json.dumps(pose_list, indent=4)
            with open(file_path, 'w') as outfile:
                outfile.write(json_object)

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
        for controller in self.gravity_comp_controllers:
            state = ControllerState()
            state.name = controller
            state.state = state.RUNNING
            goal.updates.append(state)

        ## Loop over the list of controllers that should not be in gravity compensation mode
        for controller in self.non_gravity_comp_controllers:
            state = ControllerState()
            state.name = controller
            state.state = state.STOPPED
            goal.updates.append(state)
        
        ## Send the assembled goal to the action server responsible for controller state management
        self.controller_client.send_goal(goal)


    def record_ee_trajectory(self):
        '''
        A function that records the human-guided motions

        Parameters:
        - self: The self reference.
        '''
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


if __name__ == '__main__':
    ## Initialize the `relax_arm_control` node
    rospy.init_node('armcontrol')

    ## Instantiate the `ArmControl()` object
    obj = ArmControl()

    ## 
    rospy.spin()
