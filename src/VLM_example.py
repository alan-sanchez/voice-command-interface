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

class TaskGenerationAndSpeechToText:
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

        self.bridge = CvBridge()
        self.sub = rospy.Subscriber('/head_camera/rgb/image_raw', Image, self.callback, queue_size=1)
    
        ## Specify the relative path from the home directory and construct the full path using the user's home directory
        self.relative_path = 'catkin_ws/src/voice_command_interface'
        # self.image_path = os.path.join(os.environ['HOME'], self.relative_path, 'images/vlm.jpeg')

        ## Open the file in read mode
        prompt = os.path.join(os.environ['HOME'], self.relative_path, 'prompts/action_list.txt')
        
        with open(prompt, 'r') as file:
            ## Read the contents of the file into a string
            self.initial_prompt = file.read()

    def callback(self,img_msg):
        self.image = img_msg

    def generate_response(self,max_length=1000):
        '''

        '''
        # Convert image
        try:
            image = self.bridge.imgmsg_to_cv2(self.image, 'bgr8')
        except CvBridgeError as e:
            rospy.logwarn('CV Bridge error: {0}'.format(e))
        rospy.loginfo("Saving Image now")

        ## Save image
        completeName = os.path.join(os.environ['HOME'], self.relative_path, 'images/VLM.jpeg')
        cv2.imwrite(completeName, image)

        with open(completeName, "rb") as image_file:
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
                    "text": "You are a robot that assists with bartending. What’s in this image? If there is a bottle, how much liquid is in left in it? Can you provide the pixel bounding box for each item?"
                    },
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                    }
                ]
                }
            ],
            "max_tokens": max_length
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        print(content)



if __name__ == "__main__":
    ## Initialize the `` node
    rospy.init_node('vlm_example_local')

    ## Instantiate the `TaskGenerationAndSpeechToText` class
    combined_class = TaskGenerationAndSpeechToText()
    
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
    
