#!/usr/bin/env python

## Import modules
import rospy
import sys
import signal
import json
import actionlib
import moveit_commander 
import moveit_msgs.msg

## Import message types and other python libraries
from moveit_python import PlanningSceneInterface, MoveGroupInterface
from moveit_msgs.msg import MoveItErrorCodes
from robot_controllers_msgs.msg import QueryControllerStatesAction, QueryControllerStatesGoal, ControllerState

# Define a class attribute or a global variable as a flag
interrupted = False

# Define a signal handler function
def signal_handler(signal, frame):
    global interrupted
    interrupted = True

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)

class Record:
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

        # Create the list of joints
        self.joints = ["torso_lift_joint", "shoulder_pan_joint", "shoulder_lift_joint", "upperarm_roll_joint",
                  "elbow_flex_joint", "forearm_roll_joint", "wrist_flex_joint", "wrist_roll_joint"]
        

        controller_states = "/query_controller_states"

        self._controller_client = actionlib.SimpleActionClient(controller_states,QueryControllerStatesAction)
        self._controller_client.wait_for_server()

        self._gravity_comp_controllers = ["arm_controller/gravity_compensation"]

        self._non_gravity_comp_controllers = list()
        self._non_gravity_comp_controllers.append("arm_controller/follow_joint_trajectory")
        self._non_gravity_comp_controllers.append("arm_with_torso_controller/follow_joint_trajectory")

    def init_pose(self, vel = 0.2):
        """
        Function that sends a joint goal that moves the Fetch's arm and torso to
        the initial position.
        :param self: The self reference.
        :param vel: Float value for arm velocity.
        """
        ## List of joint values of init position
        pose =[.15, 1.41, 0.30, -0.22, -2.25, -1.56, 1.80, -0.37,]
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
        '''
        goal = QueryControllerStatesGoal()

        for controller in self._gravity_comp_controllers:
            state = ControllerState()
            state.name = controller
            state.state = state.RUNNING
            goal.updates.append(state)

        for controller in self._non_gravity_comp_controllers:
            state = ControllerState()
            state.name = controller
            state.state = state.STOPPED
            goal.updates.append(state)

        self._controller_client.send_goal(goal)

    def record(self):
        global interrupted
        pose_arr = []
       
        while not interrupted:
            joints_arr = self.group.get_current_joint_values()
            # Check if all joint values are not zero before appending
            if not all(value == 0 for value in joints_arr):
                pose_arr.append(joints_arr)
            rospy.sleep(1)
        
        
        file_name = raw_input('\nname your movement file: ')

        dictionary = {
            "poses": pose_arr,
        }

        json_object = json.dumps(dictionary, indent=4)
        with open(file_name + '.json', 'w') as outfile:
            outfile.write(json_object)
        
        # Reset interrupted flag
        interrupted = False
        
if __name__ == '__main__':
    ## Initialize the `relax_arm_control` node
    rospy.init_node('record')

    ## Instantiate the `ArmControl()` object
    obj = Record()

    # raw_input("Press enter to move Fetch to it's initial arm configuration")
    # obj.init_pose()
    # print("")
    
    # obj.relax_arm()
    # raw_input("My arm is in relax mode. You can move it and hover it above the center top of the object you want me to disinfect. Once you have done that, press Enter.")
    
    # raw_input("I will start recording the cleaning task once you press enter")
    obj.record()
    ## Notify user that they can move the arm
    # rospy.loginfo("Relaxed arm node activated. You can now move the manipulator")
    # print()
    rospy.loginfo("Type Ctrl + C when you are done recording")
    rospy.spin()