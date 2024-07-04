#!/usr/bin/env python3
import os
import argparse
import sounddevice as sd
import soundfile as sf
from openai import OpenAI

class TextToSpeech:
    """
    A class that handles text to speech conversion and audio playback using OpenAI API.
    """
    def __init__(self):
        """
        Initializes the TextToSpeech class with API key and sets up the client.
        """        
        ## Specify the relative and images directory path
        self.relative_path = 'catkin_ws/src/voice_command_interface/audio_files'        
        
        ## Get the OpenAI API key from environment variables
        self.key = os.environ.get("openai_key")
        
        ## Initialize the OpenAI client with the API key
        self.client = OpenAI(api_key=self.key)

    def speech(self, text='Hello World!', filename="default"):
        """
        Converts text to speech and saves it to a file.
        
        Parameters:
        text (str): The text to convert to speech.
        filename (str): The filename to save the speech audio.
        """
        ## Create a speech response using OpenAI's API
        response = self.client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text
        )

        # Construct the full file path for saving the audio file and save the response
        file_dir = os.path.join(os.environ['HOME'], self.relative_path, filename + ".wav")         
        response.stream_to_file(file_dir)

    def playback(self, filename):
            """
            Plays back the specified audio file if it exists.
            
            Parameters:
            filename (str): The filename of the audio file to play back.
            """
            ## Construct the full file path for the audio file
            file_dir = os.path.join(os.environ['HOME'], self.relative_path, filename + ".wav")
            
            # Check if the file exists, if so, then play it back
            if os.path.exists(file_dir):
                data, fs = sf.read(file_dir, dtype='float32')
                sd.play(data, fs)
                status = sd.wait()
                print(f"Playing {file_dir}")
            else:
                ## Notify the user if the file does not exist
                print(f"File {file_dir} does not exist.")
           

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Convert text to speech using OpenAI API.')
    # parser.add_argument('text', type=str, help='The text to convert to speech')
    # parser.add_argument('filename', type=str, help='The filename to save the speech audio')
    # args = parser.parse_args()

    ## Create an instance of the TextToSpeech class
    tts = TextToSpeech()
    # tts.speech(args.text, args.filename)

    ## Run speech function with user inputs
    tts.speech("Hello, Skinny Human?", "intro")

    ## Playback audio
    tts.playback("intro")


