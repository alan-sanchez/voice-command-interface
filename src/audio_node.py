#!/usr/bin/env python3

import rospy
import sys
import os

import sounddevice as sd
import soundfile as sf

from scipy.io.wavfile import write
from pathlib import Path
from openai import OpenAI
from std_msgs.msg import String

class AudioCommunication:
    """
    A class that handles audio communication using ROS and OpenAI API.
    """
    def __init__(self):
        """
        Initializes the AudioCommunication class, sets up subscribers, paths, API client, and recording parameters.
        Parameters:
        - self: The self reference. 
        """
        ## Initialize subscriber
        # self.sub = rospy.Subscriber('/speech_recognition/final_result', String, self.callback, queue_size=10)
        self.pub = rospy.Publisher('/talk', String, queue_size=10)
        
        ## Specify the relative and images directory path
        self.relative_path = 'catkin_ws/src/voice_command_interface/'
        self.intro_audio_dir = os.path.join(os.environ['HOME'], self.relative_path, 'audio_files/intro.wav')
        self.voice_recording_dir = os.path.join(os.environ['HOME'], self.relative_path, 'audio_files/voice_record.wav')
        
        ## Get the OpenAI API key from environment variables
        self.key = os.environ.get("openai_key")
        
        ## Initialize the OpenAI client with the API key
        self.client = OpenAI(api_key=self.key)

        ## Recording parameters: sampling rate and duration
        self.fs = 44100 # Sampling rate in Hz
        self.seconds = 6 # Recording duration in seconds

        ## Log initialization notifier
        rospy.loginfo('{}: is ready.'.format(self.__class__.__name__))

    def initial_interaction(self):
        """
        Callback function that gets called when a new message is received on the subscribed topic.
        Parameters:
        - self: The self reference.
        - msg (String): The string message coming in from initial greeting
        """
        ## Begin greeting
        data, fs = sf.read(self.intro_audio_dir, dtype='float32')
        sd.play(data, fs)
        status = sd.wait() # Wait until playback is finished
        self.pub.publish("start")
            
        ## Begin voice recording
        ## Record audio from the microphone for the specified duration
        myrecording = sd.rec(int(self.seconds * self.fs), samplerate=self.fs, channels=2)
        sd.wait()  # Wait until recording is finished

        ## Save the recorded audio as a WAV file
        write(self.voice_recording_dir, self.fs, myrecording)  # Save as WAV file 
    
    # def suggestion(self):





if __name__ == '__main__':
    ## Initialize the ROS node with the name 'audio_communication'
    rospy.init_node('audio_communication', argv=sys.argv)

    ## Create an instance of the AudioCommunication class
    obj = AudioCommunication()

    input("Press Enter when ready.")

    obj.initial_interaction()

    ## Keep the program running and listening for callbacks
    rospy.spin()