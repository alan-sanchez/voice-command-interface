#!/usr/bin/env python3

import rospy
import sys
import os
import json
import ast

import sounddevice as sd
import soundfile as sf

from gpt_features import SpeechToText, TextToText, TextToSpeech
from scipy.io.wavfile import write
from pathlib import Path
from openai import OpenAI
from std_msgs.msg import String
from halo import Halo
from geometry_msgs.msg import Pose

class AudioNode():
    """
    A class that handles audio communication using ROS and OpenAI API.
    """
    def __init__(self):
        """
        Initialize the AudioNode class with ROS subscribers, publishers, and OpenAI API key and client. 
        """
        ## Initialize subscriber
        self.object_map_sub = rospy.Subscriber('object_map', String, self.callback, queue_size=10)
        
        ## Initialize publisher
        self.known_obj_pub         = rospy.Publisher('known_object_dict',    String, queue_size=10)
        self.unknown_obj_pub       = rospy.Publisher('unknown_object_label', String, queue_size=10)
        self.cleaning_status_pub   = rospy.Publisher('cleaning_status',      String, queue_size=10)
        self.human_demo_status_pub = rospy.Publisher('human_demo_status',    String, queue_size=10)
        
        ## Specify the relative, prompt, and audio directory paths
        self.relative_path = 'catkin_ws/src/voice_command_interface/'
        self.system_filename = 'system_prompt.txt'
        self.system_filename_dir = os.path.join(os.environ['HOME'], self.relative_path, 'prompts', self.system_filename)

        ## Get the OpenAI API key from environment variables
        self.key = os.environ.get("openai_key")
        
        ## Initialize the OpenAI client with the API key
        self.client = OpenAI(api_key=self.key)

        ## Initialize SpeechToText, TextToText, and TextToSpeech classes
        self.stt = SpeechToText()
        self.ttt = TextToText()
        self.tts = TextToSpeech()

        ## Recording parameters: sampling rate and duration
        self.fs = 44100  # Sampling rate in Hz
        self.default_time = 10 # Duration in seconds

        ## hard code line number
        self.line_number = 38

        # Initialize object map dictionary and label list
        # self.object_map_dict = {'rum bottle':    {'centroid': [0.663, -0.095, 1.22],  'status': 'contaminated', 'table_height': 0.78, 'in_repo':False},
        #                         'red solo cup':  {'centroid': [0.621, 0.202, 0.863],  'status': 'clean',        'table_height': 0.78, 'in_repo':True}, 
        #                         'squirt soda':   {'centroid': [0.628, -0.124, 0.955], 'status': 'contaminated', 'table_height': 0.78, 'in_repo':True}
        #                         }
        self.label_list =[]

        ## Initialize Halo spinner for indicating processing
        self.spinner = Halo(text='Computing response', spinner='dots')

        ## Log initialization notifier
        rospy.loginfo('{}: is ready.'.format(self.__class__.__name__))
        

    def callback(self, msg):
        '''
        Callback function for the 'object_map' topic. Updates the object map dictionary
        and modifies the system prompt file with the received message.

        Parameters:
        - msg (str): dictionary of object_map in a string format.     
        '''
        print('made it here')
        ## Load the object map dictionary from the received message
        self.object_map_dict = json.loads(msg.data)
        self.label_list = list(self.object_map_dict.keys())

        ## Read the file contents
        with open(self.system_filename_dir, 'r') as file:
            lines = file.readlines()

        ## Delete the specified line
        del lines[self.line_number-1]
        lines.insert(self.line_number - 1, msg.data + '\n')

        ## Write the modified contents back to the file
        with open(self.system_filename_dir, 'w') as file:
            file.writelines(lines)


    def record_audio(self, filename=None):
        """
        Method to record audio, convert it to text, and get a response from OpenAI API.

        Parameters:
        - system_filename (None or String): the filename of the system prompt. If set to None, the
          text_to_text method will set a default system prompt. 
   
        Returns:
        -Response(dictionary): reponse of GPT in a dictionary format. 
        """
        ##
        input("\nPress Enter to start recording")
        
        ## Record audio from the microphone
        self.myrecording = sd.rec(int(self.default_time * self.fs), samplerate=self.fs, channels=2)
        input('Recording... Press Enter to stop.\n')  # Wait for the user to press Enter to stop the recording
        sd.stop()
        
        ## Start the spinner to indicate processing
        self.spinner.start()

        ## create temporary file name and save the recorded audio as a WAV file
        temp_filename = 'temp_recording.wav'
        write(temp_filename, self.fs, self.myrecording)  # Save as WAV file 
        
        ## Use whisper speech to text (stt) converter
        transcript = self.stt.convert_to_text(temp_filename)
        os.remove(temp_filename)

        ## 
        response = self.ttt.text_to_text(system_filename=filename, user_content=transcript)
        
        ## Return the response as a dictionary
        return ast.literal_eval(response)
    
    
    def get_task(self, dict_response):
        '''
        Process the response dictionary and execute tasks based on the keys.

        Parameters:
        - dict_repsonse (dictionary): dictionary returned from gpt.
        '''        
        ## Extract key from the response dictionary (Assuming the dictionary length size is 1)
        key_list = list(dict_response.keys())
        
        ## Filter out clean objects from dictionary
        filtered_dict = {key: value for key, value in self.object_map_dict.items() if value['status'] != 'clean'}
        
        ## Extract the values (labels) for the contaminated objects
        value_list = list(dict_response.values())

        ## Greeting condition
        if key_list[0] == 'A':
            self.spinner.stop()
            self.tts.playback('intro.wav') # Audio was created before the developement of this script

        ## Known objects condition
        elif key_list[0] == 'B':
            self.spinner.stop()
            if len(filtered_dict) != 0:
                # Publish the dictionary as a String
                self.contaminated_obj_pub.publish(str(filtered_dict))
                self.tts.playback('disinfection_notification.wav') # Audio was created before the developement of this script
            else:
                ## Use playback to notify user that no items were moved
                self.tts.playback('no_contamination.wav') # Audio was created before the developement of this script
                
        # Unknown objects condition
        elif key_list[0] == 'C':
            ## Pull lables of unknown objects
            unkown_obj_names = [item for sublist in value_list for item in sublist]

            ## Create question for human operator
            message = f"{', '.join(unkown_obj_names)} is not in my repository of known items. Would you like to show me how by guiding my arm? Or would you rather I disinfect items that are in my repository?"

            ## Convert message to speech and play back for human operator
            self.tts.convert_to_speech(message, 'unknown.wav')
            self.spinner.stop()
            self.tts.playback('unknown.wav')

            ## Record response of human operator
            response = self.record_audio(filename='new_trajectory.txt')
            self.spinner.stop()
            
            ## 
            for key, value in response.items():
                if value == 1:
                    ##
                    self.cleaning_status_pub.publish("cleaning")

                    input("\nPress Enter to relax arm")
        
                    self.human_demo_status_pub.publish('relax')

                    input("\n Press Enter to start recording")

                    self.human_demo_status_pub.publish('start')

                    input("\n Press Enter to stop recording")

                    self.human_demo_status_pub.publish('finish')

                    input("\n Press Enter to move to Init Pose")

                    self.human_demo_status_pub.publish('init_pose')
                    self.cleaning_status_pub.publish("complete")
                else: 
                    ##
                    known_item_dict = {key: value for key, value in filtered_dict.items() if value['in_repo'] == True}
                    self.contaminated_obj_pub.publish(str(known_item_dict))
            print(response)
            

if __name__ == '__main__':
    ## Initialize the ROS node with the name 'audio_communication'
    rospy.init_node('audio_communication', argv=sys.argv)

    ## Create an instance of the AudioCommunication class
    obj = AudioNode()

    # rospy.spin()
    while True:
        dict_response = obj.record_audio()
        task = obj.get_task(dict_response)
        # print(dict_response)
