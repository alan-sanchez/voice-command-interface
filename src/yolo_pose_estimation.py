#!/usr/bin/env python3

import cv_bridge
import numpy as np
import roslib.packages
import rospy
from sensor_msgs.msg import Image
from ultralytics import YOLO
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from ultralytics_ros.msg import YoloResult

class PoseEstimation():
    """

    """
    def __init__(self):
        """
        
        """
        self.model = YOLO(rospy.get_param("yolo_model"))

        self.results_sub = rospy.Subscriber("yolo_result", YoloResult, self.callback)
        
        rospy.loginfo('{}: is ready.'.format(self.__class__.__name__))


    def callback(self,msg):
        # msg.detections
        # result_keys = [key for key, value in self.model.items() if value in ]
        # return result_keys

        # for ids in msg.detections.detections:
            
        # print(len(msg.detections.detections))

        # for res in msg.detections.detections:
        #     print([res.bbox.center.x, res.bbox.center.y])
            
        # id_list = [result.id for detection in msg.detections.detections for result in detection.results]
        
        # Assuming bbox.center.x and bbox.center.y are attributes of your objects
        # Collecting id values and bounding box centers into a dictionary
        result_dict = {result.id: [detection.bbox.center.x, detection.bbox.center.y] for detection in msg.detections.detections for result in detection.results}

        # Now result_dict contains ids as keys and corresponding bounding box centers as values
        print(result_dict)

        
if __name__=="__main__":
    # Initialize irradiance_vectors node
    rospy.init_node('yolo_pose_estimation',anonymous=True)

    # Instantiate the `PoseEstimation` class
    PoseEstimation()
    rospy.spin()


    
