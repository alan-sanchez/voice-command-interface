#!/usr/bin/env python3

## Import needed libraries
import cv2
import signal
import rospy
import message_filters
import os
import tf
import time

import numpy as np
import matplotlib.pyplot as plt
import pyransac3d as pyrsc
import sensor_msgs.point_cloud2 as pc2

from cluster import Cluster
from bounding_boxes import BBox
from sensor_msgs.msg import PointCloud2, Image, PointCloud
from geometry_msgs.msg import Point32
from cv_bridge import CvBridge

## Define a class attribute or a global variable as a flag
interrupted = False

## Function to handle signal interruption
def signal_handler(signal, frame):
    global interrupted
    interrupted = True

## Assign the signal handler to the SIGINT signal
signal.signal(signal.SIGINT, signal_handler)

class ObjectSegmentation(Cluster, BBox):
    '''
    '''
    def __init__(self):
        '''
        Constructor method for initializing the ObjectSegmentation class.

        Parameters:
        - self: The self reference.
        '''
        Cluster.__init__(self,pixel_size=0.005, dilation_size=6) # Call the inherited classes constructors
        BBox.__init__(self)
        ## Specify the relative and images directory path
        self.relative_path = 'catkin_ws/src/voice_command_interface/images/'
        self.image_directory = os.path.join(os.environ['HOME'], self.relative_path)

        ##
        # self.pub = rospy.Publisher('/test', PointCloud, queue_size=1)

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
        self.list_yolo_msgs = []
        self.pcl2_msg = None
        self.counter  = 0
        self.y_buffer = 15 # number of pixels to buffer in y direction
        
        ## Log initialization notifier
        rospy.loginfo('{}: is ready.'.format(self.__class__.__name__))

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

    def segment_image(self):
        '''
        Function that segments all the object in the image.

        Parameters:
        - self: The self reference.
        '''
        # Initialize a new point cloud message type to store position data.
        temp_cloud = PointCloud()
        temp_cloud.header = self.pcl2_msg.header

        ## Convert the image message to an OpenCV image format using the bridge 
        raw_image = self.bridge.imgmsg_to_cv2(self.img_msg, desired_encoding='bgr8')
        
        x = []
        y = []
        z = []
        # x_coordinates = []
        # y_coordinates = []

        ## For loop to extract pointcloud2 data into a list of x,y,z, and store it in a pointcloud message (pcl_cloud)
        for data in pc2.read_points(self.pcl2_msg, skip_nans=True):
            temp_cloud.points.append(Point32(data[0],data[1],data[2]))

        transformed_cloud = self.transform_pointcloud(temp_cloud, target_frame="base_link")

        for point in transformed_cloud.points:
            x.append(point.x)
            y.append(point.y)
            z.append(point.z)
        
        arr = np.column_stack((x,y,z))
        plane1 = pyrsc.Plane()
        best_eq, _ = plane1.fit(arr, 0.01)
        table_height = abs(best_eq[3]) +.03
        # print(best_eq)

        data = []        
        for i, zeta in enumerate(z):
            if zeta > table_height:
                # x_coordinates.append(x[i])
                # y_coordinates.append(y[i])
                data.append((float(x[i]), float(y[i]), float(z[i])))
        
        data_arr = np.array(data)
        regions_dict = self.fit(data_arr)
        bboxes = self.compute_bbox(regions_dict)
        for k, bbox in enumerate(bboxes):
            cropped_image = raw_image[bbox[1]:bbox[3], bbox[0]:bbox[2]] #y_min, y_max, x_min, x_max
            img_name = 'cropped_image_' + str(k) + '.jpeg'
            temp_directory = os.path.join(os.environ['HOME'], self.relative_path, img_name)
            cv2.imwrite(temp_directory, cropped_image)
        
        # plt.figure(figsize=(20,20))
        # plt.plot(x_coordinates, y_coordinates, "xk", markersize=14)        
        # plt.show()
       

        ############################# Vision To Text #################################
        # start_time = rospy.Time.now()

        # text_responses = self.generate_response(img=raw_image, bboxes=bbox_list)
        
        # for response in text_responses:
        #     print(response)

        # computation = rospy.Time.now() - start_time
        # # time_conversion = 
        # print("Computation time: " + str(computation.to_sec()))     
        


        # ###
        # plt.figure(figsize=(20,20))
        # plt.imshow(raw_image) 
        # plt.scatter(x_coord,y_coord, color='b', marker='x', s=100)
        # plt.axis('off')
        # plt.show()

        # plt.figure(figsize=(20,20))
        # plt.plot(x_coordinates, y_coordinates, "xk", markersize=14)        
        # plt.show()

    
    
    def transform_pointcloud(self,pcl_msg, target_frame):
        """
        Function that ...
        :param self: The self reference.
        :param msg: The PointCloud message.

        :returns new_cloud: The transformed PointCloud message.
        """
        while not rospy.is_shutdown():
            try:
                new_cloud = self.listener.transformPointCloud(target_frame ,pcl_msg)
                return new_cloud
            except (tf.LookupException, tf.ConnectivityException,tf.ExtrapolationException):
                pass
    
                
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


