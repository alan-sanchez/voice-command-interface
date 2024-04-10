#!/usr/bin/env python3

## Import needed libraries
import cv2
import numpy as np
import roslib.packages
import rospy
import message_filters
# import json
import os
import tf
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Image
from ultralytics import YOLO
# from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
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

        ## Specify the relative path from the home directory and construct the full path using the user's home directory
        relative_path = 'catkin_ws/src/voice_command_interface/'
        model_directory      = os.path.join(os.environ['HOME'], relative_path, 'models/mobilenet_v3_small_075_224_embedder.tflite')
        self.image_directory = os.path.join(os.environ['HOME'], relative_path, 'images/')


        ## Create options for Image Embedder
        base_options = python.BaseOptions(model_asset_path=model_directory)
        l2_normalize = True #@param {type:"boolean"}
        quantize = True #@param {type:"boolean"}
        self.options = vision.ImageEmbedderOptions(base_options=base_options, 
                                                   l2_normalize=l2_normalize, 
                                                   quantize=quantize)

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
        self.raw_image_sub    = message_filters.Subscriber("/head_camera/rgb/image_raw", Image)

        ## Synchronize YOLO results and depth image messages       
        sync = message_filters.ApproximateTimeSynchronizer([self.yolo_results_sub,
                                                            self.image_depth_sub,
                                                            self.raw_image_sub],
                                                            queue_size=10,
                                                            slop=0.1)
        sync.registerCallback(self.callback_sync)        
        
        ## Define height end effector cushion (should be around 0.15m)
        self.height_cushion = 0.15

        ## Log initialization notifier
        rospy.loginfo('{}: is ready.'.format(self.__class__.__name__))

    def callback_srv(self,request):
        rospy.loginfo("Received Request")
        
        return CoordinatesResponse(self.objects_and_coordinates)

    def callback_sync(self,yolo_msg, depth_msg, raw_msg):
        '''
        Callback function for synchronized YOLO results and depth image messages.

        Parameters:
        - self: The self reference.
        - yolo_msg (YoloResult): Image information of the detected objects in Yolo.
        - img_msg (Image): Image message type from the Primesense camera. 
        '''
        ## Convert ROS Image message to OpenCV image
        cv_depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='16UC1')  
        cv_raw_image   = self.bridge.imgmsg_to_cv2(raw_msg, desired_encoding='bgr8')

        ## Dictionary to store coordinate information for each detected object    
        coord_dict = {}

        ## Loop through YOLO detections and compute pose and bounding box coordinates
        for detection in yolo_msg.detections.detections:
            bbox_x_center = int(detection.bbox.center.x)
            bbox_y_center = int(detection.bbox.center.y)
            bbox_width_size   = int(detection.bbox.size_x)
            bbox_height_size  = int(detection.bbox.size_y)

            coordinate_estimation, cropped_dimensions = self.compute_coordinates_and_size(bbox_x_center,
                                                                                          bbox_y_center,
                                                                                          bbox_width_size,
                                                                                          bbox_height_size,
                                                                                          cv_depth_image,
                                                                                          depth_msg.header)

            for result in detection.results:
                ## Include id name search below:
                obj = self.model.names[result.id]

                ## add key and value to dictionary               
                coord_dict[obj] = [[round(coordinate_estimation.point.x,2),
                                    round(coordinate_estimation.point.y,2),
                                    round(coordinate_estimation.point.z,2)],
                                    cropped_dimensions]
                
        ## append the dictionary to list 
        self.list_of_dictionaries.append(coord_dict)

        ## Increment counter by 1
        self.message_count += 1

        ## Use conditional statement to stop after 10 callbacks
        if self.message_count == 5:
            ## Find the largest pose_dict in the list_of_dictionaries
            dict_obj_and_bbox = (max(self.list_of_dictionaries, key=len))
            # print(dict_obj_and_bbox)
            # print(self.objects_and_coordinates, type(self.objects_and_coordinates))

            self.image_embedding(cv_raw_image, dict_obj_and_bbox)
            # for key, value in dict_obj_and_bbox.values():
            #     # print(key, value) 
    
            #     cropped_image = cv_raw_image[value[2]:value[3], value[0]:value[1]]
            #     # print(corners.keys(),corners.value())
            
            # file_name = 'camera_image.jpeg'
            # completeName = os.path.join(self.output_directory, file_name)
            # cv2.imwrite(completeName, cropped_image)

            # print(self.objects_and_coordinates)
            self.message_count = 0
            self.list_of_dictionaries[:]=[]

    def image_embedding(self,image, dict):
        '''
        
        '''

        for key, value in dict.items():
            # print(key,value)
            if key == 'cup':
                # print(value[0],value[1])
            
                ## Create Image Embedder
                with vision.ImageEmbedder.create_from_options(self.options) as embedder:
                    cup_reference  = os.path.join(self.image_directory, 'cup_reference.jpeg')
                    cropped_image  = image[value[1][2]:value[1][3], value[1][0]:value[1][1]]
                    temp_directory = os.path.join(self.image_directory, 'image.jpeg')
                    cv2.imwrite(temp_directory, cropped_image)
                    # Format images for MediaPipe
                    first_image = mp.Image.create_from_file(cup_reference)
                    second_image = mp.Image.create_from_file(temp_directory)
                    first_embedding_result = embedder.embed(first_image)
                    second_embedding_result = embedder.embed(second_image)



                    # Calculate and print similarity
                    similarity = vision.ImageEmbedder.cosine_similarity(first_embedding_result.embeddings[0],
                                                                    second_embedding_result.embeddings[0])
                    
                if similarity > 0.45:
                    print("It's a purple cup")
                else:
                    print("It's not a purple cupt")

    def compute_coordinates_and_size(self,bbox_x_center, bbox_y_center, bbox_width, bbox_height, cv_image, header):
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
        ## 
        x_min = int(bbox_x_center - (bbox_width/2))
        x_max = int(bbox_x_center + (bbox_width/2))
        y_min = int(bbox_y_center - (bbox_height/2))
        y_max = int(bbox_y_center + (bbox_height/2))
        cropped_dimensions = [x_min,x_max,y_min,y_max]
        
        ## Extract depth value from the depth image at the center of the bounding box
        z = float(cv_image[bbox_y_center][bbox_x_center])/1000.0 # Depth values are typically in millimeters, convert to meters
        
        ## Calculate x and y coordinates using depth information and camera intrinsics
        x = float((bbox_x_center - self.CX_DEPTH) * z / self.FX_DEPTH)
        y = float((bbox_y_center - self.CY_DEPTH - (bbox_height/2)) * z / self.FY_DEPTH) + self.height_cushion

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
                return transformed_point, cropped_dimensions

            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                pass
        
                
if __name__=="__main__":
    ## Initialize irradiance_vectors node
    rospy.init_node('yolo_coord_estimation',anonymous=True)

    ## Instantiate the `CoordinateEstimation` class
    CoordinateEstimation()
    rospy.spin()
