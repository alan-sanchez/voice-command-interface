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
        '''
        ## Intialize subscriber
        self.unknown_object_sub    = rospy.Subscriber('unknown_object_label', String, self.label_callback)
        self.human_demo_status_sub = rospy.Subscriber('human_demo_status',    String, self.operations_callback)
       
        ## Contrct the path to save recorded movements
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

        ## Initialize demo status
        self.demo_status = None

        ## Log initialization notifier
        rospy.loginfo('{}: is ready.'.format(self.__class__.__name__))

    def label_callback(self, msg):
        '''
        Callback function for handling incoming labels for unknown objects.

        Parameters:
        - self: The self reference.
        - msg (String): Message containing labels of the unknown objects. 
        '''
        print(msg.data)
        # self.unknown_item_dict = json.loads(msg.data)#ast.literal_eval(msg.data)
        self.label = msg.data
       
    
    def operations_callback(self,msg):
        '''
        Callback function for handling operations based on the human demonstration status.

        Parameters:
        - self: The self reference.
        - msg (String): Status of the human demo.
        '''
        print(msg.data)
        if msg.data == 'relax':
            self.relax_arm()
        else:
            self.demo_status = msg.data

    
    def timer_callback(self, event):
        '''
        Timer callback function to periodically record end-effector positions.

        Parameters:
        - self: The self reference. 
        - event: The event timer.
        '''
        pose_list = []
        
        ## Continuously record poses while the demo status is 'start'
        while self.demo_status == 'start':
            pose = self.ee_pose()
            if any(value != 0 for value in pose):
                pose_list.append(pose)
            rospy.sleep(1)

        ## Save recorded poses to a file if there are any
        if len(pose_list) > 1:
            ## Construct the full file path
            file_path = os.path.join(self.full_path, str(self.label) + '.json')

            ## Save the poses to a JSON file
            json_object = json.dumps(pose_list, indent=4)
            with open(file_path, 'w') as outfile:
                outfile.write(json_object)


    def relax_arm(self):
        '''
        Turns on gravity compensation controller and turns off other controllers.
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


    def ee_pose(self):
        '''
        Function thatfinds the pose of the `gripper_link` relative to the `base_link` frame.
        Return:
        - trans + rot (list): The translation and rotation of the end-effector.
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

    ## Keep the node running
    rospy.spin()
