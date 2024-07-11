#!/usr/bin/env python3

## Import python libraries
import os
import base64
import requests
import rospy
import sys
import os
import cv2
import signal

# from openai import OpenAI
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

## Define a class attribute or a global variable as a flag
interrupted = False

## Function to handle signal interruption
def signal_handler(signal, frame):
    global interrupted
    interrupted = True

## Assign the signal handler to the SIGINT signal
signal.signal(signal.SIGINT, signal_handler)

class VisionToText:
    '''
    A class that combines task generation and speech-to-text functionality.
    '''
    def __init__(self, model = "gpt-4o"):
        '''
        A constructor that initializes the tokenizer and model.

        Parameters:
        - self: The self reference.
        - model_id (str): large language model identifier.
        '''
        ## Pull and set key for openai
        self.model_id = model
        self.key = os.environ.get("openai_key")

        ## Create a `CVBridge` object
        self.bridge = CvBridge()
    
        ## Specify the relative path from the home directory and construct the full path using the user's home directory
        self.relative_path = 'catkin_ws/src/voice_command_interface/'

        ## 
        default_image_dir  = os.path.join(os.environ['HOME'], self.relative_path, 'images/vlm.jpeg')
        
        ## Convert the image message to an OpenCV image format using the bridge
        image_bgr = cv2.imread(default_image_dir)
        self.default_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) 
        # self.ex_image = self.bridge.imgmsg_to_cv2(image_rgb, desired_encoding='bgr8')

    def viz_to_text(self,img = 'default_img', prompt="what do you see?", bbox=[0, 0, 640, 480], max_length=1000):
        '''
        A function that ...
        Reference: https://platform.openai.com/docs/guides/vision

        Parameters:
        - img (Image)
        - bboxes (list)
        - prompt (str)

        '''
        if isinstance(img, str): 
            img = self.default_image


        # text_list = [] 
        # for k, bbox in enumerate(bboxes):
        cropped_image = img[bbox[1]:bbox[3], bbox[0]:bbox[2]] #y_min, y_max, x_min, x_max
        img_name = 'temp.jpeg'
        temp_directory = os.path.join(os.environ['HOME'], self.relative_path, 'images',img_name)
        cv2.imwrite(temp_directory, cropped_image)

        with open(temp_directory, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.key}"
            }

            payload = {
                "model": self.model_id,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": 
                            {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        },
                    ]
                    }
                ],
                "max_tokens": max_length
            }

            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            #    text_list.append(content)
        
        os.remove(temp_directory)
        return content # text_list


if __name__ == "__main__":
    ## Initialize the `` node
    rospy.init_node('vision_to_text')

    ## Instantiate the `TaskGenerationAndSpeechToText` class
    combined_class = VisionToText()
    
    ## outer loop controlling create movement or play movement
    while True:
        print("\n\nEnter 1 for Fetch to describe what it sees. \nEnter 2 to quit\n")
        control_selection = input("Choose an option: ")

        ## sub loop controlling add a movement feature
        if control_selection == "1":
            response_text = combined_class.viz_to_text()
            print(response_text)
        
        elif control_selection == "2":
            break
        
        else:
            print("\nInvalid selection\n")
    