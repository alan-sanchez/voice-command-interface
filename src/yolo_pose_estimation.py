#!/usr/bin/env python3

## Import needed libraries
import cv2
import numpy as np
import roslib.packages
import rospy
import message_filters

from sensor_msgs.msg import Image
from ultralytics import YOLO
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from ultralytics_ros.msg import YoloResult
from cv_bridge import CvBridge, CvBridgeError

class PoseEstimation():
    """
    Class for estimating poses using YOLO and Primesense camera.         
    """
    def __init__(self):
        """
        Constructor method for initializing the PoseEstimation class.

        Parameters:
        - self: The self reference.
        """
        ## YOLO model initialization
        self.model = YOLO(rospy.get_param("yolo_model"))

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
                                                            queue_size=5,
                                                            slop=0.2)

        sync.registerCallback(self.callback_sync)        
        
        ## Log initialization notifier
        rospy.loginfo('{}: is ready.'.format(self.__class__.__name__))


    def callback_sync(self,yolo_msg, img_msg):
        """
        Callback function for synchronized YOLO results and depth image messages.

        Parameters:
        - self: The self reference.
        - yolo_msg (YoloResult): Image information of the detected objects in Yolo.
        - img_msg (Image): Image message type from the Primesense camera. 
        """
        ## Convert ROS Image message to OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='16UC1')  

        ## Dictionary to store pose information for each detected object    
        pose_dict = {}

        ## Loop through YOLO detections and compute pose for each
        for detection in yolo_msg.detections.detections:
            bbox_x_center = int(detection.bbox.center.x)
            bbox_y_center = int(detection.bbox.center.y)
            
            pose = self.compute_pose(bbox_x_center,bbox_y_center,cv_image)

            for result in detection.results:
                pose_dict[result.id] = pose
        
        print(pose_dict)
        # ## Collecting id values and bounding box centers into a dictionary
        # result_dict = {result.id: [detection.bbox.center.x, detection.bbox.center.y] for detection in yolo_msg.detections.detections for result in detection.results}

        # ## Now result_dict contains ids as keys and corresponding bounding box centers as values
        # print(result_dict)

    def compute_pose(self, bbox_x_center, bbox_y_center,cv_image):
        """
        Compute 3D pose based on bounding box center and depth image.

        Parameters:
        - self: The self reference.
        - bbox_x_center (int): The x center location of the bounding box.
        - bbox_y_center (int): The y center location of the bounding box.
        - cv_image (image): OpenCV image of the headcamera data.

        Return:
        - [x,y,z] (list): The pose estimation location of the detected object.
        """
        z = cv_image[bbox_y_center][bbox_x_center]
        x = (bbox_x_center - self.CX_DEPTH) * z / self.FX_DEPTH
        y = (bbox_y_center - self.CY_DEPTH) * z / self.FY_DEPTH
        return [x,y,z]
    
        
if __name__=="__main__":
    ## Initialize irradiance_vectors node
    rospy.init_node('yolo_pose_estimation',anonymous=True)

    ## Instantiate the `PoseEstimation` class
    PoseEstimation()
    rospy.spin()


    
