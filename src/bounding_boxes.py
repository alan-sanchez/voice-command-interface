#!/usr/bin/env python3

## Import needed libraries
import cv2
import numpy as np
import rospy
import os
import tf

from geometry_msgs.msg import Point32
from sensor_msgs.msg import PointCloud

class BBox():
    '''
    Class to create bounding boxes for detected objects.  
    '''
    def __init__(self):
        '''
        Constructor method for initializing the BBox class.

        Parameters:
        - self: The self reference.
        '''   
        ## Initialize TransformListener class
        self.listener = tf.TransformListener()
    
        ## Primsense Camera Parameters
        self.FX_DEPTH = 529.8943482259415
        self.FY_DEPTH = 530.4502363904782
        self.CX_DEPTH = 323.6686505142122
        self.CY_DEPTH = 223.8369337605044

        ## Bounding box margin
        self.pixel_margin = 20 #number of pixels

    def compute_bbox(self, region_dict):
        '''
        Function that computes bounding box from the camera's physical parameters.
        reference link: https://stackoverflow.com/questions/31265245/
        Parameters:
        - self: The self reference.
        - region_dict (dictionary): Dictionary regions in an image.
        
        Return:
        - bboxes (list): A list containing the bounding boxes of the detected objects.
        '''        
        ## Create a list to store the bounding box values for each detected object
        bboxes = []

        ## Run for loop 
        for key in region_dict:
            ## Initialize a PointCloud message
            point_cloud_msg = PointCloud()
            point_cloud_msg.header.stamp = rospy.Time.now()
            point_cloud_msg.header.frame_id = "base_link"  # Set the appropriate frame

            ## Iterate through each (x,y,z) tuple in the list for the current key
            for x, y, z in region_dict[key]:
                ## Create a Point32 message for each (x,y,z) coordinate
                point = Point32()
                point.x = x
                point.y = y
                point.z = z 

                ## Append the point to the PointCloud message
                point_cloud_msg.points.append(point)
            
            ## Transform the point cloud to the target frame
            transformed_cloud = self.transform_pointcloud(point_cloud_msg, "head_camera_rgb_optical_frame")   
            
            ## Initialize lists to store the transformed x, y coordinates values
            x_img = []
            y_img = []

            ## Iterate through each point in the transformed point cloud
            for point in transformed_cloud.points:
                ## Calculate the depth, x, and y coordinate in the image plane
                depth = point.z * 1000 
                u = int((1000 * point.x * self.FX_DEPTH / depth) + self.CX_DEPTH)
                v = int((1000* point.y * self.FY_DEPTH / depth) + self.CY_DEPTH)
                x_img.append(u)                
                y_img.append(v)
            
            ## Calculate the minimum and maximum x and y coordinates with pixel margin
            x_min = min(x_img) - self.pixel_margin
            x_max = max(x_img) + self.pixel_margin
            y_min = min(y_img) - self.pixel_margin
            y_max = max(y_img) + self.pixel_margin

            ## Constrain x_min and y_min to be at least 0
            x_min = max(0, x_min)
            y_min = max(0, y_min)

            ## Constrain x_max to be at most 640 and y_max to be at most 480
            x_max = min(640, x_max)
            y_max = min(480, y_max)

            ## Append the calculated bounding box to the bboxes list 
            bboxes.append([x_min, y_min, x_max, y_max])

        return bboxes    


    def transform_pointcloud(self,pcl_msg, target_frame):
        """
        Function that transforms the PointCloud to the desired target_frame.
        Parameters:
        - self: The self reference.
        - pcl_msg(PointCloud): The point cloud of the detected objects.

        Returns:
        - new_cloud(PointCloud): The transformed PointCloud message.
        """
        while not rospy.is_shutdown():
            try:
                new_cloud = self.listener.transformPointCloud(target_frame ,pcl_msg)
                return new_cloud
            except (tf.LookupException, tf.ConnectivityException,tf.ExtrapolationException):
                pass
    
                
if __name__=="__main__":
    ## Initialize irradiance_vectors node
    # rospy.init_node('yolo_coord_estimation',anonymous=True)

    ## Instantiate the `CoordinateEstimation` class
    obj = BBox()
