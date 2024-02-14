#!/usr/bin/env python3

## Import needed libraries
import cv2
import numpy as np
import roslib.packages
import rospy
import message_filters
import json
import os
import tf

from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Image
from ultralytics import YOLO
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from ultralytics_ros.msg import YoloResult
from cv_bridge import CvBridge, CvBridgeError

class PoseEstimation():
    '''
    Class for estimating poses using YOLO and Primesense camera.         
    '''
    def __init__(self):
        '''
        Constructor method for initializing the PoseEstimation class.

        Parameters:
        - self: The self reference.
        '''
        ## YOLO model initialization
        yolo_model = rospy.get_param("yolo_model")
        path = roslib.packages.get_pkg_dir("voice_command_interface")
        self.model = YOLO(f"{path}/models/{yolo_model}")
        # print(type(self.model.names))

        ## 
        self.listener = tf.TransformListener()

        ## 
        self.list_of_dictionaries = []

        ## 
        self.message_count = 0

        ## 
        self.target_frame = "base_link"

        ##
        self.output_directory = "/home/alan/catkin_ws/src/voice_command_interface/src"

        ## ROS bridge for converting between ROS Image messages and OpenCV images
        self.bridge =CvBridge()
    
        ## Primsense Camera Parameters
        self.FX_DEPTH = 529.8943482259415
        self.FY_DEPTH = 530.4502363904782
        self.CX_DEPTH = 323.6686505142122
        self.CY_DEPTH = 223.8369337605044
        
        ## Subscribers for YOLO results and depth image from the camera
        self.yolo_results_sub = message_filters.Subscriber("yolo_result", YoloResult)
        self.image_depth_sub  = message_filters.Subscriber("head_camera/depth_registered/image", Image)

        ## Synchronize YOLO results and depth image messages       
        sync = message_filters.ApproximateTimeSynchronizer([self.yolo_results_sub,
                                                            self.image_depth_sub],
                                                            queue_size=10,
                                                            slop=0.4)

        sync.registerCallback(self.callback_sync)        
        
        ## Log initialization notifier
        rospy.loginfo('{}: is ready.'.format(self.__class__.__name__))


    def callback_sync(self,yolo_msg, img_msg):
        '''
        Callback function for synchronized YOLO results and depth image messages.

        Parameters:
        - self: The self reference.
        - yolo_msg (YoloResult): Image information of the detected objects in Yolo.
        - img_msg (Image): Image message type from the Primesense camera. 
        '''
        ## Convert ROS Image message to OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='16UC1')  

        ## Dictionary to store pose information for each detected object    
        pose_dict = {}

        ## 
        self.header = img_msg.header
        print(self.header)
        print(yolo_msg.header)
        # transformation_matrix = self.get_transform_matrix(self.target_frame, img_msg.header)


        ## Loop through YOLO detections and compute pose for each
        for detection in yolo_msg.detections.detections:
            bbox_x_center = int(detection.bbox.center.x)
            bbox_y_center = int(detection.bbox.center.y)
            
            camera_pose_estimation = self.compute_pose(bbox_x_center,bbox_y_center,cv_image,img_msg.header)
            print(camera_pose_estimation.point)
            print()
        #     # Check if all elements in pose are zero
        #     if all(element == 0 for element in camera_pose_estimation):
        #         continue
        #     else:
        #         camera_pose_estimation = np.append(camera_pose_estimation, [[1]],axis=0)

        #     # print(camera_pose_estimation)
        #     # print(transformation_matrix)
        #     # print()
        #     # Perform matrix multiplication to transform the coordinates
        #     # if camera_pose_estimation is not None:
        #     transformed_coordinates = np.dot(transformation_matrix, camera_pose_estimation)

        #     # print(transformed_coordinates)
            
        #     for result in detection.results:
        #         ## Include search below:
        #         obj = self.model.names[result.id]

        #         pose_dict[obj] = [transformed_coordinates[0][0],
        #                           transformed_coordinates[1][0],
        #                           transformed_coordinates[2][0]]

        #         # pose_dict[obj] = [camera_pose_estimation[0][0],
        #         #                   camera_pose_estimation[1][0],
        #         #                   camera_pose_estimation[2][0]]

        # ##
        # self.list_of_dictionaries.append(pose_dict)

        # ##
        # self.message_count += 1

        # ## 
        # if self.message_count == 10:
        #     ## Find the largest pose_dict in the list_of_dictionaries
        #     largest_pose_dict = max(self.list_of_dictionaries, key=len)
        #     print(largest_pose_dict)
        #     # print()

        #     # self.message_count = 0
        #     ## Save the largest pose_dict to a JSON file
        #     self.save_to_json(largest_pose_dict)

    
    def compute_pose(self,bbox_x_center, bbox_y_center, cv_image, header):
        '''
        Compute 3D pose based on bounding box center and depth image.

        Parameters:
        - self: The self reference.
        - bbox_x_center (int): The x center location of the bounding box.
        - bbox_y_center (int): The y center location of the bounding box.
        - cv_image (image): OpenCV image of the headcamera data.
        - header (header):

        Return:
        - [x,y,z] (list): The pose estimation location of the detected object.
        '''
        z = float(cv_image[bbox_y_center][bbox_x_center])
        x = float((bbox_x_center - self.CX_DEPTH) * z / self.FX_DEPTH)
        y = float((bbox_y_center - self.CY_DEPTH) * z / self.FY_DEPTH)
        print(x,y,z)
        point = PointStamped()
        point.header=header
        point.point.x = x 
        point.point.y = y 
        point.point.z = z 

        while not rospy.is_shutdown():
            try:
                transformed_pose = self.listener.transformPoint('/base_link', point)
                return transformed_pose

            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                pass
        



    # def compute_pose(self, bbox_x_center, bbox_y_center,cv_image):
    #     '''
    #     Compute 3D pose based on bounding box center and depth image.

    #     Parameters:
    #     - self: The self reference.
    #     - bbox_x_center (int): The x center location of the bounding box.
    #     - bbox_y_center (int): The y center location of the bounding box.
    #     - cv_image (image): OpenCV image of the headcamera data.

    #     Return:
    #     - [x,y,z] (list): The pose estimation location of the detected object.
    #     '''
    #     z = float(cv_image[bbox_y_center][bbox_x_center])
    #     x = float((bbox_x_center - self.CX_DEPTH) * z / self.FX_DEPTH)
    #     y = float((bbox_y_center - self.CY_DEPTH) * z / self.FY_DEPTH)
    #     arr = np.array([[x],[y],[z]])
    #     return arr

    # def get_transform_matrix(self,target_frame, hdr):
    #     """
    #     Get the transform matrix from the source frame to the target frame.

    #     Parameters:
    #     - source_frame (str): The source frame.
    #     - target_frame (str): The target frame.
    #     - timestamp (rospy.Time): The timestamp for the transform (optional).

    #     Returns:
    #     - transform_matrix (list): The 4x4 transform matrix.
    #     """
    #     while not rospy.is_shutdown():
    #         try:
    #             transform_matrix = self.listener.asMatrix(target_frame,hdr)
    #             return transform_matrix

    #         except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
    #             rospy.logerr(f"Error getting transform: {e}")
    #             return None

    def save_to_json(self, data):
        '''

        '''
        # Ensure the specified directory exists
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)

        # Save the data to a JSON file in the specified directory
        json_file_path = os.path.join(self.output_directory, 'largest_pose_dict.json')
        with open(json_file_path, 'w') as json_file:
            json.dump(data, json_file, indent=2)
    
        
if __name__=="__main__":
    ## Initialize irradiance_vectors node
    rospy.init_node('yolo_pose_estimation',anonymous=True)

    ## Instantiate the `PoseEstimation` class
    PoseEstimation()
    rospy.spin()
