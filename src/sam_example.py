#!/usr/bin/env python3

## Import python libraries
import rospy
import torch
import cv2
import signal
import os
import numpy as np
import matplotlib.pyplot as plt

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

class ItemSegmentation:
    '''
    
    '''
    def __init__(self):
        '''
        
        '''
        ## Log initialization notifier
        self.relative_path = 'catkin_ws/src/voice_command_interface/'
        model_path = os.path.join(os.environ['HOME'], self.relative_path, 'models/sam_vit_h_4b8939.pth')
        self.image_path = os.path.join(os.environ['HOME'], self.relative_path, 'images/VLM.jpeg')
     
        ##
        DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        ##
        sam = sam_model_registry["vit_h"](checkpoint=model_path)
        sam.to(device=DEVICE)
        
        ## 
        self.mask_generator = SamAutomaticMaskGenerator(model=sam,
                                                        points_per_side=8,
                                                        pred_iou_thresh=0.98,
                                                        stability_score_thresh=0.98,
                                                        )

        rospy.loginfo('{}: has initialized.'.format(self.__class__.__name__))

    def segment_image(self):
        '''
        
        '''
        image_bgr = cv2.imread(self.image_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        # cropped_image = image_rgb[45:250,170:610] #[y_min, y_max, x_min, x_max]
        
        masks = self.mask_generator.generate(image_rgb)
        max_area = -1  # Start with a value that's less than any possible area value

        ## 
        for i, mask in enumerate(masks):
            if mask['area'] > max_area:
                max_area = mask['area']
                index = i
        ##
        table_bbox = self.get_vertices(masks[index]['bbox'])
        table_polygon = Polygon(table_bbox)
        
        ##
        for j, mask in enumerate(masks):
            if j == index:
                continue
            
            ##
            bbox = self.get_vertices(masks[j]['bbox'])

            ##
            for x, y in bbox:
                point = Point(x,y)
                if table_polygon.contains(point):
                    cropped_image = image_rgb[bbox[0][1]:bbox[2][1], bbox[0][0]:bbox[2][0]]
                    img_name = 'cropped_image_' + str(j) + '.jpeg'
                    temp_directory = os.path.join(os.environ['HOME'],self.relative_path, 'images',img_name)
                    cv2.imwrite(temp_directory, cropped_image)

                    print("Item is on the table")
                    break


        plt.figure(figsize=(20,20))
        plt.imshow(image_rgb) #cropped_image)
        self.show_anns(masks)
        plt.axis('off')
        plt.show() 


    def get_vertices(self, bbox):
        '''
        Extracts the bounding box vertices from a mask at a given index.

        Parameters:
        - masks: List of dictionaries, each containing a 'bbox' key with values in XYWH format.
        - index: Integer index of the specific mask in the list from which to extract coordinates.

        Returns:
        - ADD HERE.
        '''
        # Extract values from the bbox and calculate x and y coordinates
        x_min = bbox[0]
        y_min = bbox[1]
        x_max = x_min + bbox[2]
        y_max = y_min + bbox[3]

        # Return the coordinates of each vertex 
        return [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]


    def show_anns(self,anns):
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)

        img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
        img[:,:,3] = 0
        for ann in sorted_anns:
            m = ann['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.35]])
            img[m] = color_mask
        ax.imshow(img)

    def show_mask(self, mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)
        
    def show_points(self, coords, labels, ax, marker_size=375):
        pos_points = coords[labels==1]
        neg_points = coords[labels==0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
        
    def show_box(self, box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

if __name__=="__main__":
    ## Initialize irradiance_vectors node
    rospy.init_node('yolo_coord_estimation',anonymous=True)

    ## Instantiate a `CoordinateEstimation` object
    obj= ItemSegmentation()

    ## outer loop controlling create movement or play movement
    while True:
        print("\n\nEnter 1 for Fetch to segment what it sees. \nEnter 2 to quit\n")
        control_selection = input("Choose an option: ")

        ## sub loop controlling add a movement feature
        if control_selection == "1":
            obj.segment_image()
        elif control_selection == "2":
            break



