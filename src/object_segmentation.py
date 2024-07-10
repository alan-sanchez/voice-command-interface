#!/usr/bin/env python3

## Import needed libraries
import cv2
import signal
import rospy
import message_filters
import os
import tf
import copy

import numpy as np
import matplotlib.pyplot as plt
import pyransac3d as pyrsc
import sensor_msgs.point_cloud2 as pc2

from cluster import Cluster
from vision_to_text import VisionToText
from bounding_boxes import BBox

from sensor_msgs.msg import PointCloud2, Image, PointCloud
from geometry_msgs.msg import Point32
from cv_bridge import CvBridge
from std_msgs.msg import String

# ## Define a class attribute or a global variable as a flag
# interrupted = False

# ## Function to handle signal interruption
# def signal_handler(signal, frame):
#     global interrupted
#     interrupted = True

# ## Assign the signal handler to the SIGINT signal
# signal.signal(signal.SIGINT, signal_handler)

class ObjectSegmentation(Cluster, VisionToText, BBox):
    '''

    '''
    def __init__(self):
        '''
        Constructor method for initializing the ObjectSegmentation class.
        '''
        ## Call the inherited classes constructors
        Cluster.__init__(self,pixel_size=0.005, dilation_size=6)
        VisionToText.__init__(self) 
        BBox.__init__(self)

        ##
        self.sub = rospy.Subscriber('/talk', String, self.callback, queue_size=10)

        ## Specify the relative and images directory path
        self.relative_path = 'catkin_ws/src/voice_command_interface/'
        self.image_directory = os.path.join(os.environ['HOME'], 'images/', self.relative_path)
        label_prompt_dir = os.path.join(os.environ['HOME'], self.relative_path, 'prompts/', 'label_prompt.txt')
        updated_map_dir = os.path.join(os.environ['HOME'], self.relative_path, 'prompts/', 'updated_map.txt')

        ##
        with open(label_prompt_dir, 'r') as file:
            self.label_prompt = file.read()

        ## 
        with open(updated_map_dir, 'r') as file:
            self.updated_prompt = file.read()
        
        ## Subscribe and synchronize YOLO results, PointCloud2, and depth image messages
        self.pcl2_sub         = message_filters.Subscriber("/head_camera/depth_downsample/points", PointCloud2)
        self.raw_image_sub    = message_filters.Subscriber("/head_camera/rgb/image_raw", Image)
        sync = message_filters.ApproximateTimeSynchronizer([self.pcl2_sub,
                                                            self.raw_image_sub],
                                                            queue_size=1,
                                                            slop=1.2)
        sync.registerCallback(self.callback_sync) 

        ## Initialize TransformListener class
        self.listener = tf.TransformListener(True, rospy.Duration(10.0)) 

        ## ROS bridge for converting between ROS Image messages and OpenCV images
        self.bridge =CvBridge()      
        
        ## Initialize variables and constants
        self.pcl2_msg = None
        self.table_height_buffer = 0.03 # in meters
        self.max_table_reach = 0.85 # in meters
        self.object_location_dict = {}
        self.threshold = .010 # in meters

        ## Log initialization notifier
        rospy.loginfo('{}: is ready.'.format(self.__class__.__name__))

    def callback(self, str_msg):
        '''
        
        '''
        if str_msg == "start":
            dict = self.segment_image(visual_debugger=False)
            self.label_images(dict)

            self.timer = rospy.Timer(rospy.Duration(5), self.timer_callback)

        # elif str_msg == "help":


    def timer_callback(self, event):
        self.location_updater()

    def callback_sync(self, pcl2_msg, img_msg):
        '''
        Callback function to store synchronized messages.

        Parameters:
        - self: The self reference.
        - pcl2_msg (PointCloud2): Point cloud information from the Primesense camera. 
        - img_msg (Image): Image message type from the Primesense camera. 
        '''
        self.pcl2_msg = pcl2_msg
        self.img_msg  = img_msg


    def segment_image(self, visual_debugger=True):
        '''
        Function that segments all the object in the image.

        Parameters:
        - self: The self reference.
        - visual_debugger (bool): Boolean used for the visual debugging process.

        Returns:
        - 
        '''
        # Initialize a new point cloud message type to store position data.
        temp_cloud = PointCloud()
        temp_cloud.header = self.pcl2_msg.header

        ## Convert the image message to an OpenCV image format using the bridge
        raw_image = self.bridge.imgmsg_to_cv2(self.img_msg, desired_encoding='bgr8')
        
        ## 
        x = []
        y = []
        z = []
        x_plot = []
        y_plot = []

        ## For loop to extract pointcloud2 data into a list of x,y,z, and store it in a pointcloud message (pcl_cloud)
        for data in pc2.read_points(self.pcl2_msg, skip_nans=True):
            temp_cloud.points.append(Point32(data[0],data[1],data[2]))

        ## Transform pointcloud to reference the base_link and store coordinates values in 3 separate lists 
        transformed_cloud = self.transform_pointcloud(temp_cloud, target_frame="base_link")
        for point in transformed_cloud.points:
            x.append(point.x)
            y.append(point.y)
            z.append(point.z)
        
        ## 
        arr = np.column_stack((x,y,z))
        plane1 = pyrsc.Plane()
        best_eq, _ = plane1.fit(arr, 0.01)
        table_height = abs(best_eq[3]) + self.table_height_buffer
        # print(best_eq)

        ## Pull coordinates of items on the table
        object_pcl_coordinates = []        
        for i in range(len(z)):
            if (z[i] > table_height) and (x[i] < self.max_table_reach):
                x_plot.append(x[i]) # used for visual debugger
                y_plot.append(y[i]) # used for visual debugger
                object_pcl_coordinates.append([float(x[i]), float(y[i]), float(z[i])])
        object_pcl_coordinates = np.array(object_pcl_coordinates)
        
        ## Create region dictionary
        regions_dict = self.compute_regions(object_pcl_coordinates)

        ## 
        bbox_list = []
        for id in regions_dict:
            bbox = self.compute_bbox(regions_dict[id])
            bbox_list.append(bbox) # Also used for visual debugger
            regions_dict[id]["bbox"] = bbox
            del regions_dict[id]["points"]


        ## Conditional statement to visualize the segmented images and the x and y coordinates 
        if visual_debugger:
            for k, bbox in enumerate(bbox_list):
                cropped_image = raw_image[bbox[1]:bbox[3], bbox[0]:bbox[2]] #y_min, y_max, x_min, x_max
                img_name = 'cropped_image_' + str(k) + '.jpeg'
                temp_directory = os.path.join(os.environ['HOME'], self.relative_path, 'images', img_name)
                cv2.imwrite(temp_directory, cropped_image)

            # plt.figure(figsize=(20,20))
            # plt.plot(x_plot, y_plot, "xk", markersize=14)        
            # plt.show()
            os.remove(temp_directory)
        return regions_dict 
    

    def label_images(self,regions_dict):
        '''
        A function that ...

        Parameters:
        - self: The self reference.
        - regions_dict(dictionary): 
        '''
        ## Convert the image message to an OpenCV image format using the bridge 
        raw_image = self.bridge.imgmsg_to_cv2(self.img_msg, desired_encoding='bgr8')

        for id in regions_dict:
            label = self.generate_response(img=raw_image, prompt=self.label_prompt, bbox=regions_dict[id]["bbox"])
            self.object_location_dict[label] = regions_dict[id]["centroid"]
        
        print(self.object_location_dict)


    def location_updater(self):
        '''

        '''
        ## 
        current_data_dict = self.segment_image()
        copy_obj_dict = self.object_location_dict.copy()
        
        ##
        for label, centroid in self.object_location_dict.items():
            for key in current_data_dict:
                euclidean_dist = np.linalg.norm(np.array(centroid) - np.array(current_data_dict[key]["centroid"]))
                
                if euclidean_dist < self.threshold:
                    current_data_dict.pop(key)
                    copy_obj_dict.pop(label)
                    break
       
        print()
        ##
        if len(copy_obj_dict) == 1 and len(current_data_dict) == 1:
            label, = copy_obj_dict
            print(f"The {label} has moved. updating map")
            id = next(iter(current_data_dict))
            self.object_location_dict[label] = current_data_dict[id]["centroid"].copy()

        ## 
        elif len(copy_obj_dict) > 1 and len(current_data_dict) > 1: # for id in current_data_dict:
            keys = copy_obj_dict.keys()
            labels = list(keys)
            updated_prompt = self.updated_prompt + str(labels)

            ## Convert the image message to an OpenCV image format using the bridge 
            raw_image = self.bridge.imgmsg_to_cv2(self.img_msg, desired_encoding='bgr8')


            # if len(copy_obj_dict) < len(current_data_dict):
                

            for id in current_data_dict:
                label = self.generate_response(img=raw_image, prompt=updated_prompt, bbox=current_data_dict[id]["bbox"])
                if label in labels:
                    self.object_location_dict[label] = current_data_dict[id]["centroid"].copy()
                else:
                    print(label)
            
            print(f"The {labels} have moved. Updating map")
    
                
if __name__=="__main__":
    ## Initialize irradiance_vectors node
    rospy.init_node('object_segmentation',anonymous=False)

    ## Instantiate the `CoordinateEstimation` class
    obj = ObjectSegmentation()

    rospy.spin()

    # print("\n\nEnter 1 for Fetch to segment what it sees. \nElse enter anything or ctrl+c to exit the node\n")
    # control_selection = input("Choose an option: ")

    # ## sub loop controlling add a movement feature
    # if control_selection == "1":
    #     dict = obj.segment_image(visual_debugger=True)
    #     obj.label_images(dict)

    #     # #
    #     while True:
    #         rospy.sleep(5)
    #         obj.location_updater()
            

    