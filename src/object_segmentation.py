#!/usr/bin/env python3

## Import needed libraries
import cv2
import signal
import rospy
import message_filters
import os
import tf
import ast
import json

import numpy as np
import matplotlib.pyplot as plt
import pyransac3d as pyrsc
import sensor_msgs.point_cloud2 as pc2

from cluster import Cluster
from bounding_boxes import BBox
from gpt_features import VisionToText

from sensor_msgs.msg import PointCloud2, Image, PointCloud
from geometry_msgs.msg import Point32
from cv_bridge import CvBridge
from std_msgs.msg import String
from halo import Halo

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
    A class that handles object segmentation using various techniques such as clustering and bounding box computations.
    '''
    def __init__(self):
        '''
        Constructor method for initializing the ObjectSegmentation class.
        '''
        ## Initialize objects for bounding box computation, clustering, text-to-text, and vision-to-text conversion
        self.bbox_obj = BBox()
        self.cluster_obj = Cluster(pixel_size=0.005, dilation_size=10)
        self.vtt_obj = VisionToText()   

        ## Define the filename for the prompt that instructs the VLM how to label objects
        self.label_prompt_filename = 'label_prompt.txt'

        ## Initialize publisher
        self.object_map_pub = rospy.Publisher('/object_map', String, queue_size=10)

        ## Specify the relative path for the txt files that handle the updated maps
        self.relative_path = 'catkin_ws/src/voice_command_interface/'
        updated_map_dir = os.path.join(os.environ['HOME'], self.relative_path, 'prompts/', 'updated_map.txt')
        label_list_dir = os.path.join(os.environ['HOME'], self.relative_path, 'prompts/', self.label_prompt_filename)

        ## Read the updated map prompt from a file
        with open(updated_map_dir, 'r') as file:
            self.updated_map_prompt = file.read()

        ## Pull the lits of items the robot knows how to disinfect
        with open(label_list_dir, 'r') as file:
            lines = file.readlines()
            last_line = lines[-1].strip()
            self.list_of_items = ast.literal_eval(last_line)
        
        ## Initialize subscribers
        self.cleaning_status_sub   = rospy.Subscriber('cleaning_status',      String, self.cleaning_status_callback)
        self.human_demo_status_sub = rospy.Subscriber('human_demo_status',    String, self.human_demo_callback)
        self.unknown_obj_sub       = rospy.Subscriber('unknown_object_label', String, self.in_repo_callback)

        self.pcl2_sub      = message_filters.Subscriber("/head_camera/depth_downsample/points", PointCloud2)
        self.raw_image_sub = message_filters.Subscriber("/head_camera/rgb/image_raw", Image)
        sync               = message_filters.ApproximateTimeSynchronizer([self.pcl2_sub,
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
        self.table_height_buffer = 0.04 # in meters
        self.table_height = 0.5 # in meters
        self.max_table_reach = 0.8 # in meters
        self.object_map_dict = {}
        self.threshold = .05 # in meters
        self.start = True # Start flag for initial run
        self.json_data = None # JSON data to be published
        self.cleaning_status = "complete"
        
        ## Initialize a timer to periodically update locations
        self.timer = rospy.Timer(rospy.Duration(2.5), self.timer_callback)

        ## Log initialization notifier
        rospy.loginfo('{}: is booting up.'.format(self.__class__.__name__))


    def in_repo_callback(self,str_msg):
        '''
        Callback function for changine the 'in_repo' status for unknown objects.

        Parameters:
        - str_msg (String): The message containing the label of the unknown object.
        '''
        self.object_map_dict[str_msg.data]['in_repo']=True



    def timer_callback(self, event):
        '''
        Callback function for the timer event.

        Parameters:
        - event: The timer event.
        '''
        if self.start == True:
            rospy.sleep(5)
            self.start = False 
        elif self.start == False and self.cleaning_status == 'complete':
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


    def cleaning_status_callback(self, str_msg):
        '''
        Callback function that notifies the status of the cleaning robot.

        Parameters:
        str_msg (String): a string message that states if the robot is cleaning or completed it's task.
        '''
        if str_msg.data == "cleaning":
            self.cleaning_status = str_msg.data
            rospy.loginfo("Cleaning items. Not updating map")
        
        elif str_msg.data == "complete":
            self.cleaning_status= str_msg.data
            rospy.loginfo("Cleaning is complete. ")
            self.location_updater()

            # Iterate through the dictionary and update 'contaminated' status to 'clean'
            for key, value in self.object_map_dict.items():
                if value['status'] == 'contaminated':
                    value['status'] = 'clean'

            ## Publish updated dictionary
            self.json_data = json.dumps(self.object_map_dict)
            self.object_map_pub.publish(self.json_data)
    

    def human_demo_callback(self, str_msg):
        '''
        Callback function for handling human demonstration status.

        Parameters:
        - str_msg (String): The message containing the status of the human demonstration.
        '''
        if str_msg.data == 'pause':
            self.cleaning_status = 'cleaning'
            rospy.loginfo("Arm is being guided. Not updating map")


    def segment_image(self, visual_debugger=False):
        '''
        Function that segments all the object in the image.

        Parameters:
        - self: The self reference.
        - visual_debugger (bool): Boolean used for the visual debugging process.

        Returns:
        - regions_dict (dict): Dictionary containing regions with bounding boxes for each detected object. 
        '''
        ## Initialize a new point cloud message type to store position data.
        temp_cloud = PointCloud()
        temp_cloud.header = self.pcl2_msg.header

        ## Convert the image message to an OpenCV image format using the bridge
        raw_image = self.bridge.imgmsg_to_cv2(self.img_msg, desired_encoding='bgr8')
        
        ## Initialize lists to store x, y, z coordinates
        x = []
        y = []
        z = []
        x_plot = []
        y_plot = []

        ## For loop to extract pointcloud2 data into a list of x,y,z, and store it in a pointcloud message (pcl_cloud)
        for data in pc2.read_points(self.pcl2_msg, skip_nans=True):
            temp_cloud.points.append(Point32(data[0],data[1],data[2]))

        ## Transform pointcloud to reference the base_link and store coordinates values in 3 separate lists 
        transformed_cloud = self.bbox_obj.transform_pointcloud(temp_cloud, target_frame="base_link")
        for point in transformed_cloud.points:
            x.append(point.x)
            y.append(point.y)
            z.append(point.z)
        
        ## Combine x, y, z coordinates into a single array and fit a plane to determine table height 
        arr = np.column_stack((x,y,z))
        plane1 = pyrsc.Plane()
        best_eq, _ = plane1.fit(arr, 0.01)
        self.table_height = round(abs(best_eq[3]) + self.table_height_buffer,2)
        # print(best_eq)

        ## Extract coordinates of items on the table
        object_pcl_coordinates = []        
        for i in range(len(z)):
            if (z[i] > self.table_height) and (x[i] < self.max_table_reach):
                x_plot.append(x[i]) # used for visual debugger
                y_plot.append(y[i]) # used for visual debugger
                object_pcl_coordinates.append([float(x[i]), float(y[i]), float(z[i])])
        object_pcl_coordinates = np.array(object_pcl_coordinates)
        
        ## Create region dictionary by clustering the object's x and y coordinates
        regions_dict = self.cluster_obj.compute_regions(object_pcl_coordinates)

        ## Compute bounding boxes for each region and update the regions dictionary
        bbox_list = []
        for id in regions_dict:
            bbox = self.bbox_obj.compute_bbox(regions_dict[id])
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
            
            # plt.figure(figsize=(10,10))
            # plt.plot(x_plot, y_plot, "xk", markersize=14)        
            # plt.show()

        ## Return the dictionary containing regions with bounding boxes for each detected object
        return regions_dict   


    def label_images(self,regions_dict):
        '''
        A function that creates labes for the segmented images.

        Parameters:
        - self: The self reference.
        - regions_dict(dictionary): 
        '''
        
        ## Convert the image message to an OpenCV image format using the bridge 
        raw_image = self.bridge.imgmsg_to_cv2(self.img_msg, desired_encoding='bgr8')

        ## Iterate over each region in the regions dictionary
        for id in regions_dict:

            ## Use VisionToText object to generate a label for the region based on the bounding box
            label = self.vtt_obj.viz_to_text(img=raw_image, bbox=regions_dict[id]["bbox"], prompt_filename = self.label_prompt_filename)
            self.object_map_dict[label] = {
                'centroid': regions_dict[id]["centroid"],
                'status': 'clean',
                'table_height': self.table_height
                }
            
            ## Check if the labeled object is in the list of known items
            if label in self.list_of_items:
                self.object_map_dict[label]['in_repo'] = True
            else:
                self.object_map_dict[label]['in_repo'] = False

        ## Log the status if it's the initial run
        if self.start:
            rospy.loginfo("{}: is up and running!".format(self.__class__.__name__))

        ## Convert the object map dictionary to JSON format and publish it      
        self.json_data = json.dumps(self.object_map_dict)
        self.object_map_pub.publish(self.json_data)

        print(self.object_map_dict.keys())


    def location_updater(self):
        '''
        Updates the locations of objects based on the segmented image data and the current object locations.
        '''
        if self.cleaning_status != "cleaning":
            ## Segment the current image to get data and create a copy of the current object location dictionary
            current_data = self.segment_image()
            copy_obj_map = self.object_map_dict.copy()
            
            ## Iterate over each object in the current location dictionary
            for label, data in self.object_map_dict.items():
                centroid = data['centroid']
                for key in current_data:
                    ## Calculate the Euclidean distance between the current object and the new detected object
                    euclidean_dist = np.linalg.norm(np.array(centroid) - np.array(current_data[key]['centroid']))
                    
                    ## If the distance is less than the threshold, consider it the same object
                    if euclidean_dist < self.threshold:
                        current_data.pop(key) # Remove the detected object from current data
                        copy_obj_map.pop(label) # Remove the label from the copy of the object location dictionary
                        break

            ## If there is exactly one object that moved, update its location in the map
            if len(copy_obj_map) == 1 and len(current_data) == 1:
                label, = copy_obj_map
                rospy.loginfo(f"The {label} has moved. updating map")
                id = next(iter(current_data))
                self.object_map_dict[label]['centroid'] = current_data[id]["centroid"].copy()
                self.object_map_dict[label]['status'] = 'contaminated'

                ## Publish the dictionary as a String
                self.json_data = json.dumps(self.object_map_dict)
                self.object_map_pub.publish(self.json_data)

            ## If there are multiple objects that moved, use the VLM to help with updating their locations in the map
            elif len(copy_obj_map) > 1 and len(current_data) > 1: 
                keys = copy_obj_map.keys()
                labels = list(keys)
                updated_prompt = self.updated_map_prompt + str(labels)

                ## Convert the image message to an OpenCV image format using the bridge 
                raw_image = self.bridge.imgmsg_to_cv2(self.img_msg, desired_encoding='bgr8')
                    
                for id in current_data:
                    ## Use the vision-to-text object to label the detected object based on the updated prompt
                    label = self.vtt_obj.viz_to_text(img=raw_image, prompt=updated_prompt, bbox=current_data[id]["bbox"])
                    
                    if label in labels:
                        self.object_map_dict[label]['centroid'] = current_data[id]["centroid"].copy()
                        self.object_map_dict[label]['status'] = 'contaminated'
                    else:
                        print(label)

                ## Publish the dictionary as a String
                self.json_data = json.dumps(self.object_map_dict)
                self.object_map_pub.publish(self.json_data)
                
                rospy.loginfo(f"The {labels} have moved. Updating map")
 
          
                
if __name__=="__main__":
    ## Initialize irradiance_vectors node
    rospy.init_node('object_segmentation',anonymous=False)

    ## Instantiate the `CoordinateEstimation` class
    obj = ObjectSegmentation()
    
    ## Allow some time for initialization
    rospy.sleep(1.5)

    ## Segment the image and get the regions dictionary
    dict = obj.segment_image(visual_debugger=True)

    ## Label the segmented images
    obj.label_images(dict)

    ## Keep the node running until it is shut down
    rospy.spin()
