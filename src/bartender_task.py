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

class BarTask():
    """
    A class that handles audio communication using ROS and OpenAI API.
    """
    def __init__(self):
        """
        Parameters:
        - self: The self reference. 
        """
        # self.pub = rospy.Publisher('/talk', String, queue_size=10)
        self.sub = rospy.Subscriber('object_map', String, self.callback, queue_size=10)
        
        ## Specify the relative and audio directory paths
        self.relative_path = 'catkin_ws/src/voice_command_interface/'
        self.system_filename = 'system_prompt.txt'
        self.assistant_filename = 'assistant_prompt.txt'
        self.assistant_dir = os.path.join(os.environ['HOME'], self.relative_path, 'prompts', self.assistant_filename)
        # self.intro_audio_dir = os.path.join(os.environ['HOME'], self.relative_path, 'audio_files/intro.wav')
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

        ##
        self.spinner = Halo(text='Computing response', spinner='dots')

        ## Log initialization notifier
        rospy.loginfo('{}: is ready.'.format(self.__class__.__name__))

        ## hard code line number
        self.line_number = 9


    def callback(self, msg):
        self.object_map_dict = json.loads(msg.data)
        self.label_list = list(self.object_map_dict.keys())

        # Read the file contents
        with open(self.assistant_dir, 'r') as file:
            lines = file.readlines()

        # Delete the specified line
        del lines[self.line_number-1]
        lines.insert(self.line_number - 1, str(self.label_list) + '\n')

        # Write the modified contents back to the file
        with open(self.assistant_dir, 'w') as file:
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
        response = self.ttt.text_to_text(system_filename=self.system_filename, assistant_filename=self.assistant_filename, user_prompt=transcript)
        

        ## 
        return ast.literal_eval(response)
    
    
    def get_task(self, dict_response):
        ## Assuming the dictionary is len 1
        key_list = list(dict_response.keys())
        value_list = list(dict_response.values())
        value_list = value_list[0]
        if key_list[0] == 'A':
            self.spinner.stop()
            self.tts.playback(self.audio_filename)
            
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
                    
            
            drink_list_filename = "drink_list.wav"
            self.tts.convert_to_speech(text=text, filename=drink_list_filename)
            self.spinner.stop()
            print(text)
            self.tts.playback(drink_list_filename)


        elif key_list[0] == 'C':
            print(key_list[0])

        


            

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
