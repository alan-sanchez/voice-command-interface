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
    Class for estimating poses using YOLO and Primesense camera.         
    '''
    def __init__(self):
        '''
        Constructor method for initializing the PoseEstimation class.

        Parameters:
        - self: The self reference.
        '''
        ##
        rospy.init_node('bounding_boxes',anonymous=True)

        

        ## Initialize TransformListener class
        self.listener = tf.TransformListener()
    
        ## Primsense Camera Parameters
        self.FX_DEPTH = 529.8943482259415
        self.FY_DEPTH = 530.4502363904782
        self.CX_DEPTH = 323.6686505142122
        self.CY_DEPTH = 223.8369337605044

    def compute_bbox(self, region_dict):
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
        print(region_dict.keys())

        for key in region_dict:
            # Initialize a PointCloud message
            point_cloud_msg = PointCloud()
            point_cloud_msg.header.stamp = rospy.Time.now()
            point_cloud_msg.header.frame_id = "base_link"  # Set the appropriate frame

            for x, y, z in region_dict[key]:
                point = Point32()
                point.x = x
                point.y = y
                point.z = z 

                ##
                point_cloud_msg.points.append(point)
            
            transformed_cloud = self.transform_pointcloud(point_cloud_msg, "head_camera_rgb_optical_frame")   
            x_img = []
            y_img = []
            depth = []
            for points in transformed_cloud.points:
                depth


        # for i in range(10):
            # self.pub.publish(temp_cloud)         


        # ## Extract depth value from the depth image at the center of the bounding box
        # z = float(cv_image[bbox_y_center][bbox_x_center])/1000.0 # Depth values are typically in millimeters, convert to meters
        
        # ## Calculate x and y coordinates using depth information and camera intrinsics
        # x = float((bbox_x_center - self.CX_DEPTH) * z / self.FX_DEPTH)
        # y = float((bbox_y_center - self.CY_DEPTH - (bbox_height/2)) * z / self.FY_DEPTH) + self.height_cushion

        # ## Create a PointStamped object to store the computed coordinates
        # point = PointStamped()
        # point.header=header
        # point.point.x = x 
        # point.point.y = y 
        # point.point.z = z 

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
    # rospy.init_node('yolo_coord_estimation',anonymous=True)

    ## Instantiate the `CoordinateEstimation` class
    obj = BBox()
