#!/usr/bin/env python3

## Import needed libraries
import cv2
import signal
import torch
import roslib.packages
import rospy
import message_filters
import os
import tf

import numpy as np
import matplotlib.pyplot as plt
import sensor_msgs.point_cloud2 as pc2

from sensor_msgs.msg import PointCloud2, Image, PointCloud
from geometry_msgs.msg import Point32
from ultralytics_ros.msg import YoloResult
from cv_bridge import CvBridge
from shapely.geometry import Point, Polygon 
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

## Define a class attribute or a global variable as a flag
interrupted = False

## Function to handle signal interruption
def signal_handler(signal, frame):
    global interrupted
    interrupted = True

## Assign the signal handler to the SIGINT signal
signal.signal(signal.SIGINT, signal_handler)

class ObjectSegmentation():
    '''
    Class for estimating poses using YOLO and Primesense camera.         
    '''
    def __init__(self):
        '''
        Constructor method for initializing the ObjSegmentation class.

        Parameters:
        - self: The self reference.
        '''
        ## Specify the relative and images directory path
        self.relative_path = 'catkin_ws/src/voice_command_interface/'
        self.image_directory = os.path.join(os.environ['HOME'], self.relative_path, 'images/')

        ## Segment Anything Model initiailzation
        model_path = os.path.join(os.environ['HOME'], self.relative_path, 'models/sam_vit_h_4b8939.pth')
        DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # Use computers GPU
        sam = sam_model_registry["vit_h"](checkpoint=model_path)
        sam.to(device=DEVICE)    
        self.mask_generator = SamAutomaticMaskGenerator(model=sam,
                                                        points_per_side=14,
                                                        pred_iou_thresh=0.96,
                                                        # stability_score_thresh=0.98,
                                                        ) # More information about the parameters can be found here: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/automatic_mask_generator.py#L35

        ## Subscribe and synchronize YOLO results, PointCloud2, and depth image messages
        self.yolo_results_sub = message_filters.Subscriber("/yolo_result", YoloResult)
        self.pcl2_sub         = message_filters.Subscriber("/octomap_point_cloud_centers", PointCloud2)
        self.yolo_image_sub   = message_filters.Subscriber("/yolo_image", Image)#"/head_camera/depth_downsample/points", PointCloud2)
        self.raw_image_sub    = message_filters.Subscriber("/head_camera/rgb/image_raw", Image)
        sync = message_filters.ApproximateTimeSynchronizer([self.yolo_results_sub,
                                                            self.pcl2_sub,
                                                            self.yolo_image_sub,
                                                            self.raw_image_sub],
                                                            queue_size=1,
                                                            slop=1.5)
        sync.registerCallback(self.callback_sync) 

        ## Initialize TransformListener class
        self.listener = tf.TransformListener()

        ## ROS bridge for converting between ROS Image messages and OpenCV images
        self.bridge =CvBridge()      

        ## Initialize variables and constants
        self.list_yolo_msgs = []
        self.pcl2_msg = None
        self.img_msg  = None
        self.yolo_img = None
        self.counter  = 0
        self.y_buffer = 15 # number of pixels to buffer in y direction
        
        ## Log initialization notifier
        rospy.loginfo('{}: is ready.'.format(self.__class__.__name__))

    def callback_sync(self,yolo_msg, pcl2_msg, yolo_img, img_msg):
        '''
        Callback function to store synchronized messages.

        Parameters:
        - self: The self reference.
        - yolo_msg (YoloResult): Image information of the detected objects in Yolo.
        - pcl2_msg (PointCloud2): Point cloud information from the Primesense camera. 
        - img_msg (Image): Image message type from the Primesense camera. 
        '''
        self.list_yolo_msgs.append(yolo_msg)
        self.pcl2_msg = pcl2_msg
        self.yolo_image = yolo_img
        self.img_msg  = img_msg
        self.counter += 1
        
        if self.counter > 3:
            ## Pull the longest list of detected objects from the yolo_msgs list
            max_msg_len = -1
            for msg in self.list_yolo_msgs:
                if len(msg.detections.detections) > max_msg_len:
                    self.yolo_msg = msg
        else:
            self.yolo_msg = yolo_msg
            if self.counter == 1:
                print('made it here')


    def segment_image(self):
        '''
        Function that segments all the object in the image.

        Parameters:
        - self: The self reference.
        '''
        ##     
        bbox_list = []
        yolo_polygon_list = []

        ## Loop through YOLO detections from self.yolo_msg to pull the YOLO bounding box format: [center_x, center_y, width, height] )
        for detection in self.yolo_msg.detections.detections:
            bbox_x_center = int(detection.bbox.center.x)
            bbox_y_center = int(detection.bbox.center.y)
            bbox_width_size   = int(detection.bbox.size_x)
            bbox_height_size  = int(detection.bbox.size_y)
            
            ## Converts YOLO's format to Pascal VOC's format: (x_min, y_min, x_max, y_max) 
            bbox = self.convert_bbox_from_yolo(bbox_x_center, bbox_y_center, bbox_width_size, bbox_height_size)  
            
            ## Append the bounding box to a list. Also create a polygon message type  of the bounding box for Shapely 
            ## (search if points are in the polygon) and append that polygon to a list. 
            bbox_list.append(bbox)
            yolo_polygon_list.append(Polygon(self.convert_to_poly_format(bbox)))


        ## Convert the image message to an OpenCV image format using the bridge 
        raw_image = self.bridge.imgmsg_to_cv2(self.img_msg, desired_encoding='bgr8')
        yolo_image = self.bridge.imgmsg_to_cv2(self.yolo_image, desired_encoding='bgr8')


        ## generate masks for the raw image using the SAM's mask_generator object
        masks = self.mask_generator.generate(raw_image)
        ## Run a forloop to find the index of the largest area from SAM's generated masks. The largest area, in theory, 
        ## should be the table
        max_area = -1  # Start with a value that's less than any possible area value
        for i, mask in enumerate(masks):
            if mask['area'] > max_area:
                max_area = mask['area']
                index = i  
        
        ## Convert SAM's format(x, y, width, height) to Pascal VOC's format (x_min, y_min, x_max, y_max)
        ## Then create a polygon that represnets the table.  
        table_bbox = self.convert_bbox_from_sam(masks[index]['bbox'])
        table_bbox[1] = table_bbox[1] + self.y_buffer # Add y buffer for filtering purposes
        table_polygon = Polygon(self.convert_to_poly_format(table_bbox))
        masks.pop(index) # Remove Table mask, since it isn't needed for the next step

        # table_cropped_image = raw_image[table_bbox[1]:table_bbox[3], table_bbox[0]:table_bbox[2]] #y_min, y_max, x_min, x_max
        # img_name = 'table_cropped' + '.jpeg'
        # temp_directory = os.path.join(os.environ['HOME'],self.relative_path, 'images',img_name)
        # cv2.imwrite(temp_directory, table_cropped_image)
        
        ## For loop to check if any of the polygon verticies are in the table polygon
        flag=0
        x_coord = []
        y_coord = []
        indices_to_pop = []
        for i, mask in enumerate(masks):
            ## Convert the mask's bounding box to a polygon format to get the x & y coordinates 
            ## of each vertex. Then check to see if the coordinates are in the table polygon
            temp_polygon = self.convert_to_poly_format(mask['bbox'])

            ## For loop to determine if polygon verticies (total of 4) are in the table polygon
            vertex_count = 1
            for x, y in temp_polygon:
                ## Create a Point message type for the vertex coordinates
                point = Point(x,y) 
                
                ## If bounding box's vertex is in the table polygon, then check to see if the center of that 
                ## bounding box is inside one of YOLO's bounding box. Remember that YOLO and SAM both created 
                ## a set of bounding boxes, and we want to make sure we don't detect and object twice.
                if table_polygon.contains(point):                                        
                    x_center, y_center = self.compute_bbox_center(mask['bbox'])                    
                    ## Check to see if center is inside any of YOLO's polygons (polygon_list)
                    for e, poly in enumerate(yolo_polygon_list):
                        if poly.contains(Point(x_center,y_center)):
                            ## If the center is in the polygon, then ... 
                            indices_to_pop.append(i)
                            flag = 1
                            break
                        
                        if e == (len(yolo_polygon_list)-1):
                            x_coord.append(x_center)
                            y_coord.append(y_center)
                ##
                else:
                    vertex_count += 1

                ##           
                if vertex_count == 4:
                    indices_to_pop.append(i)    
                    
                ## Break out of 2nd loop if the center of a bounding box was detected in the a polygon
                if flag == 1:
                    flag = 0
                    break
        ## Pop elements in reverse order of indices
        for index in sorted(indices_to_pop, reverse=True):
            masks.pop(index)         

        ## Initialize a new point cloud message type to store position data.
        pcl_cloud = PointCloud()
        pcl_cloud.header = self.pcl2_msg.header
        
        x_coordinates = []
        y_coordinates = []
        ## For loop to extract pointcloud2 data into a list of x,y,z, and store it in a pointcloud message (pcl_cloud)
        for data in pc2.read_points(self.pcl2_msg, skip_nans=True):
            pcl_cloud.points.append(Point32(data[0],data[1],data[2]))
            if data[2] > .785:
                x_coordinates.append(data[0])
                y_coordinates.append(data[1])


        ###
        self.counter = 0
        self.list_yolo_msgs[:] = [] 

        ####### Visual Debugger #######
        for k, ms in enumerate(masks):
            sam_bbox = self.convert_bbox_from_sam(ms['bbox'])
            cropped_image = raw_image[sam_bbox[1]:sam_bbox[3], sam_bbox[0]:sam_bbox[2]] #y_min, y_max, x_min, x_max
            img_name = 'cropped_image_' + str(k) + '.jpeg'
            temp_directory = os.path.join(os.environ['HOME'],self.relative_path, 'images',img_name)
            cv2.imwrite(temp_directory, cropped_image)

        ####
        plt.figure(figsize=(20,20))
        plt.imshow(yolo_image) 
        plt.scatter(x_coord,y_coord, color='b', marker='x', s=100)
        plt.axis('off')
        plt.show()

        plt.figure(figsize=(20,20))
        plt.scatter(x_coordinates, y_coordinates)
        plt.show()

    def convert_to_poly_format(self,bbox):
        '''
        
        '''
        x_min = int(bbox[0])
        y_min = int(bbox[1])
        x_max = int(bbox[2])
        y_max = int(bbox[3])

        #
        return [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
    
    def compute_bbox_center(self, mask_bbox):
        '''
        
        '''
        x_center = int(mask_bbox[0] + (mask_bbox[2]/2))
        y_center = int(mask_bbox[1] + (mask_bbox[3]/2))
        
        #
        return x_center,y_center

    def convert_bbox_from_sam(self, mask_bbox):
        '''
        Extracts the bounding box vertices from a mask at a given index.

        Parameters:
        - masks: List of dictionaries, each containing a 'bbox' key with values in XYWH format.
        - index: Integer index of the specific mask in the list from which to extract coordinates.

        Returns:
        - ADD HERE.
        '''
        # Extract values from the bbox and calculate x and y coordinates
        x_min = int(mask_bbox[0])
        y_min = int(mask_bbox[1])
        x_max = int(x_min + mask_bbox[2])
        y_max = int(y_min + mask_bbox[3])

        ## Return the coordinates of each vertex
        return [x_min, y_min, x_max, y_max] 

    
    def convert_bbox_from_yolo(self,bbox_x_center, bbox_y_center, bbox_width, bbox_height):
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
        y_min = int(bbox_y_center - (bbox_height/2))
        x_max = int(bbox_x_center + (bbox_width/2))
        y_max = int(bbox_y_center + (bbox_height/2))
        return [x_min, y_min, x_max, y_max] 
    
                
if __name__=="__main__":
    ## Initialize irradiance_vectors node
    rospy.init_node('object_segmentation',anonymous=True)

    ## Instantiate the `CoordinateEstimation` class
    obj = ObjectSegmentation()
    
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





# table_bounded_masks = []
        # for mask in (masks):
        #     ## Convert the mask's bounding box to a polygon format to get the x & y coordinates 
        #     ## of each vertex. Then check to see if the coordinates are in the table polygon
        #     temp_polygon = self.convert_to_poly_format(mask['bbox'])
        #     for x, y in temp_polygon:
        #         ## Create a Point message type for the vertex coordinates
        #         point = Point(x,y) 
        #         if table_polygon.contains(point):
        #             table_bounded_masks.append(mask) 
        #             break

        # print(len(table_bounded_masks))
        # for j, bounded_mask in enumerate(table_bounded_masks):
        #     x_center, y_center = self.compute_bbox_center(bounded_mask['bbox'])
        #     point = Point(x_center,y_center)
        #     x_coord.append(x_center)
        #     y_coord.append(y_center)
        #     ## Check to see if center is inside any of YOLO's polygons (polygon_list)
        #     for poly in yolo_polygon_list:
        #         if poly.contains(point):
        #             ## If the center is in the polygon, the remove that mask 
        #             table_bounded_masks.pop(j)
        #             # break                     
        # print(len(table_bounded_masks), len(x_coord))