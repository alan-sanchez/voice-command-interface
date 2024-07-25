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

class BarTask():
    """
    A class that handles audio communication using ROS and OpenAI API.
    """
    def __init__(self):
        """
        Initialize the AudioNode class with ROS subscribers, publishers, and OpenAI API key and client. 
        """
        ## Initialize subscriber
        self.object_map_sub      = rospy.Subscriber('object_map',      String, self.map_update_callback)
        self.cleaning_status_sub = rospy.Subscriber('cleaning_status', String, self.status_callback)
        
        ## Initialize publisher
        self.known_obj_pub         = rospy.Publisher('known_object_dict',    String, queue_size=10)
        self.unknown_obj_pub       = rospy.Publisher('unknown_object_label', String, queue_size=10)
        self.human_demo_status_pub = rospy.Publisher('human_demo_status',    String, queue_size=10)
        
        ## Specify the relative, prompt, and audio directory paths
        self.relative_path = 'catkin_ws/src/voice_command_interface/'
        self.cocktail_filename_dir = os.path.join(os.environ['HOME'], self.relative_path, 'prompts/cocktail_prompt.txt') 
        self.repo_filename_dir = os.path.join(os.environ['HOME'], self.relative_path, 'prompts/label_prompt.txt')

        ## Read the file contents
        with open(self.repo_filename_dir, 'r') as file:
            lines = file.readlines()
        self.repo_list = ast.literal_eval(lines[-1].strip())

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

        ## Flag for initial object list
        self.ingredient_list = []
        self.object_map_dict = {}
        self.flag = True
        self.temp_filename = 'temp_speech.wav'

        ## Initialize Halo spinner for indicating processing
        self.spinner = Halo(text='Computing response', spinner='dots')

        ## Log initialization notifier
        rospy.loginfo('{}: is ready.'.format(self.__class__.__name__))


    def append_text_to_file(self, filename, text):
        '''
        Append text to a file after removing the last line.

        Parameters:
        - filename (String): The path to the file.
        - text (String): The text to append.
        '''
        ## Read the file contents
        with open(filename, 'r') as file:
            lines = file.readlines()

        ## Read the file contents
        lines = lines[:-1]
        lines.append(text)

        ## Write the modified contents back to the file
        with open(filename, 'w') as file:
            file.writelines(lines)


    def status_callback(self, str_msg):
        '''
        Callback function for handling the status of the cleaning robot. If status is complete, notify the user

        Parameters:
        - str_msg (String): A string message indicating the status of the cleaning process.
        '''
        if str_msg.data == "complete":
            self.tts.playback("complete.wav") 


    def map_update_callback(self, msg):
        '''
        Callback function for the 'object_map' topic. Updates the object map dictionary
        and modifies the system prompt file with the received message.

        Parameters:
        - msg (String): dictionary of object_map in a string format.     
        '''        
        print()
        ## Load the object map dictionary from the received message
        self.object_map_dict = json.loads(msg.data)
        self.append_text_to_file(filename=self.cocktail_filename_dir, text= msg.data)
        contaminated_objects = {key: value for key, value in self.object_map_dict.items() if value['status'] != 'clean'} 
        
        # print(contaminated_objects)

        if len(self.ingredient_list) != 0:
            rospy.sleep(3)
            all_objs_used = all(obj in contaminated_objects for obj in self.ingredient_list)
            # print(all_objs_used)
        
            if all_objs_used and self.flag == True:
                message = "Are you finished making your " + self.drink
                self.flag = False
                self.tts.convert_to_speech(text=message, filename=self.temp_filename)
                self.tts.playback(self.temp_filename)

    def record_audio(self, filename=None):
        """
        Method to record audio, convert it to text, and get a response from OpenAI API.

        Parameters:
        - system_filename (None or String): the filename of the system prompt. If set to None, the
          text_to_text method will set a default system prompt. 
   
        Returns:
        -Response(dictionary): reponse of GPT in a dictionary format. 
        """
        ## Prompt the user to start recording
        input("Press Enter to start recording\n")
        
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

        ## Get the response from OpenAI API
        response = self.ttt.text_to_text(system_filename=filename, user_content=transcript)
        
        ## Return the response which is either a dictionary or list
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

        ############################
        ## drink selection condition
        ############################
        if key_list[0] == 'A':
            self.spinner.stop()
            self.tts.playback('ack_bartender.wav') # Audio was created before the developement of this script
            self.ingredient_list= dict_response['A']['ingredients']
            self.drink = dict_response['A']['drink']


        ##########################
        ## Known objects condition
        ##########################
        elif key_list[0] == 'B':
            self.spinner.stop()
            self.tts.playback('disinfection_notification.wav') # Audio was created before the developement of this script
            rospy.sleep(.3)
            self.known_obj_pub.publish(str(filtered_dict))
                
        ############################
        ## Unknown objects condition
        ############################
        elif key_list[0] == 'C':
            self.spinner.stop()
            
            ## Pull lables of unknown objects
            unkown_obj_names = [item for sublist in value_list for item in sublist]

            ## Create question for human operator
            message = f"{', '.join(unkown_obj_names)} is not in my repository of known items. Would you like to show me how by guiding my arm? Or would you rather I disinfect items that are in my repository?"

            ## Convert message to speech and play back for human operator
            self.tts.convert_to_speech(message, self.temp_filename)
            self.tts.playback(self.temp_filename)
            self.tts.del_speech_file(self.temp_filename)

            ## Record response of human operator
            response = self.record_audio(filename='new_trajectory.txt')
            self.spinner.stop()
            
            ## Process the response from the human operator
            for key, value in response.items():
                if value == 1:
                    self.human_demo_status_pub.publish('pause')
                    
                    ## Process each unkown object label
                    for label in unkown_obj_names:
                        self.unknown_obj_pub.publish(label)

                        ## Generate and play back message for duiing the arm
                        message = f"Okay, show me how to disifect the {label} "
                        self.tts.convert_to_speech(message, self.temp_filename)
                        self.tts.playback(self.temp_filename)

                        ## Publish relaxation status for the arm
                        self.human_demo_status_pub.publish('relax')
                        self.tts.playback('relax_arm.wav')
                        
                        ## Prompt the user to start and stop recording arm trajectory
                        input("\n Press Enter to start recording arm trajectory")
                        self.human_demo_status_pub.publish('start')
                        input("\n Press Enter to stop recording the guided path")                        
                        self.tts.playback('saving_traj.wav')
                        self.human_demo_status_pub.publish('finish')
                        
                        ## Return the arm to the initial pose
                        rospy.sleep(1)
                        self.human_demo_status_pub.publish('init_pose') 
                        rospy.sleep(6)
                        
                        ## Update the labe list with the new object
                        self.repo_list.append(label)
                        self.append_text_to_file(filename=self.repo_filename_dir, text = str(self.repo_list))
                    
                    ## Notify the user after the demonstration
                    # print(filtered_dict)
                    self.tts.playback('after_demo.wav')
                    rospy.sleep(1)
                    self.known_obj_pub.publish(str(filtered_dict))
                    
                else: 
                    ## Filter known items and publish the dictionary
                    known_item_dict = {key: value for key, value in filtered_dict.items() if value['in_repo'] == True}
                    self.known_obj_pub.publish(str(known_item_dict))

        self.flag = True
        # self.tts.del_speech_file(filename=self.temp_filename)    
        

if __name__ == '__main__':
    ## Initialize the ROS node with the name 'audio_node'
    rospy.init_node('bar_task', argv=sys.argv)

    ## Create an instance of the AudioCommunication class
    obj = BarTask()
    
    
    while True:
        dict_response = obj.record_audio('cocktail_prompt.txt')
        print(dict_response)
        task = obj.get_task(dict_response)

