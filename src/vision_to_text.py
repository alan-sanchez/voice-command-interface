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
        # self.client = OpenAI()

        self.bridge = CvBridge()
        # self.sub = rospy.Subscriber('/head_camera/rgb/image_raw', Image, self.callback, queue_size=1)
    
        ## Specify the relative path from the home directory and construct the full path using the user's home directory
        self.relative_path = 'catkin_ws/src/voice_command_interface/images'
        # self.image_path = os.path.join(os.environ['HOME'], self.relative_path)

        # ## Open the file in read mode
        # prompt = os.path.join(os.environ['HOME'], self.relative_path, 'prompts/action_list.txt')
        
        # with open(prompt, 'r') as file:
        #     ## Read the contents of the file into a string
        #     self.initial_prompt = file.read()

    # def callback(self,img_msg):
    #     self.image = img_msg

    def generate_response(self,img=None, bboxes=None, max_length=1000):
        '''
        Reference: https://platform.openai.com/docs/guides/vision
        '''
        # # Convert image
        # try:
        #     image = self.bridge.imgmsg_to_cv2(self.image, 'bgr8')
        # except CvBridgeError as e:
        #     rospy.logwarn('CV Bridge error: {0}'.format(e))
        # rospy.loginfo("Saving Image now")
        text_list = [] 
        for k, bbox in enumerate(bboxes):
            cropped_image = img[bbox[1]:bbox[3], bbox[0]:bbox[2]] #y_min, y_max, x_min, x_max
            img_name = 'temp.jpeg'
            temp_directory = os.path.join(os.environ['HOME'],self.relative_path,img_name)
            cv2.imwrite(temp_directory, cropped_image)

        ## Save image
        # completeName = os.path.join(os.environ['HOME'], self.relative_path, 'cropped_image_3.jpeg')
        # cv2.imwrite(completeName, image)

            with open(temp_directory, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.key}"
                }

                payload = {
                    "model": "gpt-4o",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                            {
                                "type": "text",
                                "text": "You are a robot that assists with bartending. There are items on a table meant to make a cocktail. Can you identify what the item is based on the image provided?"
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
                text_list.append([str(k+1) + ') ' + content])

        return text_list


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
            combined_class.generate_response()
        
        elif control_selection == "2":
            break
        
        else:
            print("\nInvalid selection\n")
    
