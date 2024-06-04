#!/usr/bin/env python3

## Import needed libraries
import cv2
import signal
import torch
import numpy as np
import roslib.packages
import rospy
import message_filters
import os
import tf
import matplotlib.pyplot as plt

from sensor_msgs.msg import PointCloud2, Image
from ultralytics_ros.msg import YoloResult
from cv_bridge import CvBridge
## SAM Stuff
from shapely.geometry import Point, Polygon
from segment_anything import  sam_model_registry, SamAutomaticMaskGenerator

## Define a class attribute or a global variable as a flag
interrupted = False

## Function to handle signal interruption
def signal_handler(signal, frame):
    global interrupted
    interrupted = True

## Assign the signal handler to the SIGINT signal
signal.signal(signal.SIGINT, signal_handler)

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
        ## Specify the relative path from the home directory and construct the full path using the user's home directory
        self.relative_path = 'catkin_ws/src/voice_command_interface/'
        self.image_directory = os.path.join(os.environ['HOME'], self.relative_path, 'images/')

        ## Segment Anything Model initiailzation
        model_path = os.path.join(os.environ['HOME'], self.relative_path, 'models/sam_vit_h_4b8939.pth')
        DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')        
        sam = sam_model_registry["vit_h"](checkpoint=model_path)
        sam.to(device=DEVICE)    
        self.mask_generator = SamAutomaticMaskGenerator(model=sam,
                                                        points_per_side=14,
                                                        pred_iou_thresh=0.91,
                                                        # stability_score_thresh=0.98,
                                                        )

        ## Subscribers for YOLO results and depth image from the camera and synchronize messages
        self.yolo_results_sub = message_filters.Subscriber("yolo_result", YoloResult)
        self.pcl2_sub         = message_filters.Subscriber("/head_camera/depth_downsample/points", PointCloud2)
        self.raw_image_sub    = message_filters.Subscriber("/head_camera/rgb/image_raw", Image)
        sync = message_filters.ApproximateTimeSynchronizer([self.yolo_results_sub,
                                                            self.pcl2_sub,
                                                            self.raw_image_sub],
                                                            queue_size=10,
                                                            slop=0.1)
        sync.registerCallback(self.callback_sync) 

        ## Initialize TransformListener class
        self.listener = tf.TransformListener()

        ## ROS bridge for converting between ROS Image messages and OpenCV images
        self.bridge =CvBridge()      

        ## Initialize variables and constants
        self.yolo_msg = None
        self.pcl2_msg = None
        self.img_msg  = None
        self.y_buffer = 10 # number of pixels
        
        ## Log initialization notifier
        rospy.loginfo('{}: is ready.'.format(self.__class__.__name__))


    def callback_sync(self,yolo_msg, pcl2_msg, img_msg):
        '''
        Callback function for synchronized YOLO results and Pointcloud2 messages.

        Parameters:
        - self: The self reference.
        - yolo_msg (YoloResult): Image information of the detected objects in Yolo.
        - img_msg (Image): Image message type from the Primesense camera. 
        '''
        self.yolo_msg = yolo_msg
        self.pcl2_msg = pcl2_msg
        self.img_msg  = img_msg
        # print("made it here")

    def segment_image(self):
        ##     
        bbox_list = []
        polygon_list = []

        ## Loop through YOLO detections and compute pose and bounding box coordinates
        for detection in self.yolo_msg.detections.detections:
            bbox_x_center = int(detection.bbox.center.x)
            bbox_y_center = int(detection.bbox.center.y)
            bbox_width_size   = int(detection.bbox.size_x)
            bbox_height_size  = int(detection.bbox.size_y)
            bbox = self.compute_yolo_bbox(bbox_x_center, bbox_y_center, bbox_width_size, bbox_height_size)   
            
            ## 
            bbox_list.append(bbox)
            polygon_list.append(Polygon(self.create_polygon(bbox)))

        ##
        raw_image = self.bridge.imgmsg_to_cv2(self.img_msg, desired_encoding='bgr8')
        masks = self.mask_generator.generate(raw_image)

        ## Compute table bounding box
        max_area = -1  # Start with a value that's less than any possible area value
        for i, mask in enumerate(masks):
            if mask['area'] > max_area:
                max_area = mask['area']
                index = i
        table_bbox = self.compute_sam_bbox(masks[index]['bbox'])
        table_bbox[2] = table_bbox[2] + self.y_buffer
        table_polygon = Polygon(self.create_polygon(table_bbox))
        # cropped_tab_image = raw_image[table_bbox[2]:table_bbox[3], table_bbox[0]:table_bbox[1]]
        # img_name = "cropped_table.jpeg"
        # temp_directory = os.path.join(os.environ['HOME'],self.relative_path, 'images',img_name)
        # cv2.imwrite(temp_directory, cropped_tab_image)

        ## 
        index_list=[]
        for j, mask in enumerate(masks):
            if j == index:
                continue
            
            ##
            polygon = self.create_polygon(masks[j]['bbox'])

            ## 
            for x, y in polygon:
                point = Point(x,y)
                if table_polygon.contains(point):
                    index_list.append(j)
                    break

        print(index_list)
        ##
        for k, ind in enumerate(index_list):
            sam_bbox = self.compute_sam_bbox(masks[ind]['bbox'])
            cropped_image = raw_image[sam_bbox[2]:sam_bbox[3], sam_bbox[0]:sam_bbox[1]] #y_min, y_max, x_min, x_max
            img_name = 'cropped_image_' + str(k) + '.jpeg'
            temp_directory = os.path.join(os.environ['HOME'],self.relative_path, 'images',img_name)
            cv2.imwrite(temp_directory, cropped_image)
            x_center, y_center = self.compute_bbox_center(masks[ind]['bbox'])#sam_bbox)
            point = Point(x_center,y_center)
            print(point)
            for poly in polygon_list:
                print(poly)
                # print(point)
                if poly.contains(point):
                    print("inside")
                    break
                else:
                    print("outside")
        
        # plt.figure(figsize=(20,20))
        # plt.imshow(raw_image) #cropped_image)
        # plt.axis('off')
        # plt.show() 


            

    

    def create_polygon(self,bbox):
        '''
        
        '''
        x_min = bbox[0]
        x_max = bbox[1]
        y_min = bbox[2]
        y_max = bbox[3]

        #
        return [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
    
    def compute_bbox_center(self, mask_bbox):
        '''
        
        '''
        x_center = mask_bbox[0] + (mask_bbox[2]/2)
        y_center = mask_bbox[1] + (mask_bbox[3]/2)
        
        #
        return x_center,y_center

    def compute_sam_bbox(self, mask_bbox):
        '''
        Extracts the bounding box vertices from a mask at a given index.

        Parameters:
        - masks: List of dictionaries, each containing a 'bbox' key with values in XYWH format.
        - index: Integer index of the specific mask in the list from which to extract coordinates.

        Returns:
        - ADD HERE.
        '''
        # Extract values from the bbox and calculate x and y coordinates
        x_min = mask_bbox[0]
        y_min = mask_bbox[1]
        x_max = x_min + mask_bbox[2]
        y_max = y_min + mask_bbox[3]

        ## Return the coordinates of each vertex
        return [x_min, x_max, y_min, y_max] #[(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]

    
    def compute_yolo_bbox(self,bbox_x_center, bbox_y_center, bbox_width, bbox_height):
        '''
        Compute 3D pose based on bounding box center and depth image.

        Parameters:
        - self: The self reference.
        - bbox_x_center (int): The x center location of the bounding box.
        - bbox_y_center (int): The y center location of the bounding box.

        Return:
        - 
        '''
        ## 
        x_min = int(bbox_x_center - (bbox_width/2))
        x_max = int(bbox_x_center + (bbox_width/2))
        y_min = int(bbox_y_center - (bbox_height/2))
        y_max = int(bbox_y_center + (bbox_height/2))
        return [x_min, x_max, y_min, y_max] #[(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
    
                
if __name__=="__main__":
    ## Initialize irradiance_vectors node
    rospy.init_node('yolo_coord_estimation',anonymous=True)

    ## Instantiate the `CoordinateEstimation` class
    obj = CoordinateEstimation()
    ## outer loop controlling create movement or play movement
    while True:
        print("\n\nEnter 1 for Fetch to segment what it sees. \nEnter 2 to quit\n")
        control_selection = input("Choose an option: ")

        ## sub loop controlling add a movement feature
        if control_selection == "1":
            obj.segment_image()
        elif control_selection == "2":
            break
        else:
            break
