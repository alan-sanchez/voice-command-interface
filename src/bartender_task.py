#!/usr/bin/env python3

import rospy
import sys
import os
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
        self.pub = rospy.Publisher('/talk', String, queue_size=10)
        
        ## Specify the relative and audio directory paths
        self.relative_path = 'catkin_ws/src/voice_command_interface/'
        self.system_filename = 'system_prompt'

        
        ## Get the OpenAI API key from environment variables
        self.key = os.environ.get("openai_key")
        
        ## Initialize the OpenAI client with the API key
        self.client = OpenAI(api_key=self.key)

        ##
        self.stt = SpeechToText()
        self.ttt = TextToText()

        ## Recording parameters: sampling rate and duration
        self.fs = 44100  # Sampling rate in Hz
        self.default_time = 10

        ##
        self.spinner = Halo(text='Computing response', spinner='dots')

        ## Log initialization notifier
        rospy.loginfo('{}: is ready.'.format(self.__class__.__name__))

    def record_audio(self):
        """
        Method to record audio.
        """
        # Record audio from the microphone
        print("Recording... Press Enter to stop.")
        self.myrecording = sd.rec(int(self.default_time * self.fs), samplerate=self.fs, channels=2)
        input()  # Wait for the user to press Enter to stop the recording
        sd.stop()
        
        self.spinner.start()

        temp_filename = "temp_recording.wav"
        ## Save the recorded audio as a WAV file
        write(temp_filename, self.fs, self.myrecording)  # Save as WAV file 
        
        ## Use whisper speech to text (stt) converter
        transcript = self.stt.convert_to_text(temp_filename)
        os.remove(temp_filename)
        
        

        response = self.ttt.text_to_text(system_filename=self.system_filename, user_prompt = transcript)
        self.spinner.stop()
        # print)
        return response


if __name__ == '__main__':
    ## Initialize the ROS node with the name 'audio_communication'
    rospy.init_node('audio_communication', argv=sys.argv)

    ## Create an instance of the AudioCommunication class
    obj = BarTask()

    while True:
        input("Press Enter to start recording")
        transcript = obj.record_audio()
        print(transcript)
