#!/usr/bin/env python3
import os
from openai import OpenAI

class SpeechToText:
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

    def convert_to_text(self, file_dir="intro.wav"):
        """
        Converts text to speech and saves it to a file.
        Link:https://platform.openai.com/docs/guides/speech-to-text
        
        Parameters:
        text (str): The text to convert to speech.
        filename (str): The filename to save the speech audio.
        """
        if file_dir == "intro.wav":
            audio_dir = os.path.join(os.environ['HOME'],self.relative_path,file_dir)

        else:
            audio_dir = file_dir

        audio_file= open(audio_dir, "rb")        
        transcription = self.client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file
        )
        # print(transcription.text)
        return(transcription.text)           

if __name__ == '__main__':
    ## Create an instance of the TextToSpeech class
    stt = SpeechToText()
    # tts.speech(args.text, args.filename)

    ## Run speech function with user inputs
    stt.convert_to_text("intro.wav")


#Landing cite reconissance
## Mission concept study, Julie and John (Systems engineer) about hoppers
## 