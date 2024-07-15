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
        Parameters:
        - self: The self reference. 
        """
        ## Initialize subscriber
        self.object_map_sub = rospy.Subscriber('object_map', String, self.callback, queue_size=10)
        
        ## Initialize publisher
        self.waypoint_pub = rospy.Publisher('waypoint', Pose, queue_size=1)
        
        ## Specify the relative and audio directory paths
        self.relative_path = 'catkin_ws/src/voice_command_interface/'
        self.system_filename = 'system_prompt.txt'
        self.system_filename_dir = os.path.join(os.environ['HOME'], self.relative_path, 'prompts', self.system_filename)
      
        ##
        self.audio_filename = 'intro.wav'

        ## Get the OpenAI API key from environment variables
        self.key = os.environ.get("openai_key")
        
        ## Initialize the OpenAI client with the API key
        self.client = OpenAI(api_key=self.key)

        ##
        self.stt = SpeechToText()
        self.ttt = TextToText()
        self.tts = TextToSpeech()

        ## Recording parameters: sampling rate and duration
        self.fs = 44100  # Sampling rate in Hz
        self.default_time = 10

        ## hard code line number
        self.line_number = 10

        ##
        self.spinner = Halo(text='Computing response', spinner='dots')

        ## Log initialization notifier
        rospy.loginfo('{}: is ready.'.format(self.__class__.__name__))
        

    def callback(self, msg):
        '''
        
        '''
        self.object_map_dict = json.loads(msg.data)
        self.label_list = list(self.object_map_dict.keys())

        # Read the file contents
        with open(self.system_filename_dir, 'r') as file:
            lines = file.readlines()

        # Delete the specified line
        del lines[self.line_number-1]
        lines.insert(self.line_number - 1, str(self.label_list) + '\n')

        # Write the modified contents back to the file
        with open(self.system_filename_dir, 'w') as file:
            file.writelines(lines)


    def record_audio(self):
        """
        Method to record audio.
        """
        # Record audio from the microphone
        self.myrecording = sd.rec(int(self.default_time * self.fs), samplerate=self.fs, channels=2)
        input('Recording... Press Enter to stop.')  # Wait for the user to press Enter to stop the recording
        sd.stop()
        
        self.spinner.start()

        temp_filename = 'temp_recording.wav'
        ## Save the recorded audio as a WAV file
        write(temp_filename, self.fs, self.myrecording)  # Save as WAV file 
        
        ## Use whisper speech to text (stt) converter
        transcript = self.stt.convert_to_text(temp_filename)
        os.remove(temp_filename)

        ## 
        response = self.ttt.text_to_text(system_filename=self.system_filename, user_prompt=transcript)
        
        ## 
        return ast.literal_eval(response)
    
    
    def get_task(self, dict_response):
        '''
        
        '''
        ## Assuming the dictionary length size is 1
        key_list = list(dict_response.keys())
        value_list = list(dict_response.values())
       
        ## Greeting condition
        if key_list[0] == 'A':
            self.spinner.stop()
            self.tts.playback(self.audio_filename)

        ## Cocktail list condition 
        elif key_list[0] == 'B':
            text = "With the options on the table, you can make a "

            for i, e in enumerate(value_list):
                if len(value_list) == 1:
                    text = text +str(e) +'.'

                else:
                    if i < len(value_list) -1:
                        text = text + str(e) + ', '
                    else:
                        text = text + "and " + str(e) + '.'
            
            ## 
            drink_list_filename = "drink_list.wav"
            self.tts.convert_to_speech(text=text, filename=drink_list_filename)
            self.spinner.stop()
            print(text)
            self.tts.playback(drink_list_filename)

        ## Manipulation condition
        elif key_list[0] == 'C':
            ##
            coordinates = self.object_map_dict.get(value_list[0])
            
            ## Create a Pose message type
            p = Pose()
            p.position.x = coordinates[0]
            p.position.y = coordinates[1]
            p.position.z = coordinates[2]
            p.orientation.x = 0.0
            p.orientation.y = 0.0
            p.orientation.z = 0.0
            p.orientation.w = 1.0

            self.waypoint_pub.publish(p)
            
            self.spinner.stop()
            print("waypoint sent")
            

if __name__ == '__main__':
    ## Initialize the ROS node with the name 'audio_communication'
    rospy.init_node('audio_communication', argv=sys.argv)

    ## Create an instance of the AudioCommunication class
    obj = BarTask()

    # rospy.spin()
    while True:
        input("\nPress Enter to start recording")
        dict_response = obj.record_audio()
        task = obj.get_task(dict_response)
        # print(dict_response)
