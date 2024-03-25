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

##
from voice_command_interface.srv import Coordinates, CoordinatesResponse 

class CoordinateEstimation():
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

        ## Create the service, specifiying the name, the service message base type
        ## and the callback function
        self.service = rospy.Service('coordinates', Coordinates, self.callback_srv)

        ## Initialize TransformListener class
        self.listener = tf.TransformListener()

        ## Initialize empty list
        self.list_of_dictionaries = []

        ## Initialize cooridnates. This will be the reponse for service call
        self.objects_and_coordinates = None

        ## Initialize counter for forloop
        self.message_count = 0

        ## Define output directory
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

    def callback_srv(self,request):
        rospy.loginfo("Received Request")
        
        return CoordinatesResponse(self.objects_and_coordinates)

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

        ## Dictionary to store coordinate information for each detected object    
        coord_dict = {}

        ## Loop through YOLO detections and compute pose for each
        for detection in yolo_msg.detections.detections:
            bbox_x_center = int(detection.bbox.center.x)
            bbox_y_center = int(detection.bbox.center.y)
            
            coordinate_estimation = self.compute_coordinates(bbox_x_center,bbox_y_center,cv_image,img_msg.header)

            for result in detection.results:
                ## Include search below:
                obj = self.model.names[result.id]

                ## add key and value to dictionary               
                coord_dict[obj] = [round(coordinate_estimation.point.x,2),
                                   round(coordinate_estimation.point.y,2),
                                   round(coordinate_estimation.point.z,2)]

        ## append the dictionary to list 
        self.list_of_dictionaries.append(coord_dict)

        ## Increment counter by 1
        self.message_count += 1

        ## Use conditional statement to stop after 10 callbacks
        if self.message_count == 10:
            ## Find the largest pose_dict in the list_of_dictionaries
            self.objects_and_coordinates = str(max(self.list_of_dictionaries, key=len))
            self.message_count = 0
            self.list_of_dictionaries[:]=[]

    def compute_coordinates(self,bbox_x_center, bbox_y_center, cv_image, header):
        '''
        Compute 3D pose based on bounding box center and depth image.

        Parameters:
        - self: The self reference.
        - bbox_x_center (int): The x center location of the bounding box.
        - bbox_y_center (int): The y center location of the bounding box.
        - cv_image (image): OpenCV image of the head camera data.
        - header (header): the header information of the head camera. 

        Return:
        - transformed_point (PointStamped): The coordinate estimation of the detected object.
        '''
        ## Extract depth value from the depth image at the center of the bounding box
        z = float(cv_image[bbox_y_center][bbox_x_center])/1000.0 # Depth values are typically in millimeters, convert to meters

        ## Calculate x and y coordinates using depth information and camera intrinsics
        x = float((bbox_x_center - self.CX_DEPTH) * z / self.FX_DEPTH)
        y = float((bbox_y_center - self.CY_DEPTH) * z / self.FY_DEPTH)

        ## Create a PointStamped object to store the computed coordinates
        point = PointStamped()
        point.header=header
        point.point.x = x 
        point.point.y = y 
        point.point.z = z 

        ## Loop until ROS shutdown, attempting to transform the point to '/base_link' frame
        while not rospy.is_shutdown():
            try:
                transformed_point = self.listener.transformPoint('/base_link', point)
                return transformed_point

            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                pass
        
                
if __name__=="__main__":
    ## Initialize irradiance_vectors node
    rospy.init_node('yolo_coord_estimation',anonymous=True)

    ## Instantiate the `CoordinateEstimation` class
    CoordinateEstimation()
    rospy.spin()
