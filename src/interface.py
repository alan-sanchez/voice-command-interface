#!/usr/bin/env python3

## Import python libraries
import gradio as gr
import whisper
import openai


class SpeechToText:
    ''' 
    A class that lanuches gradio, a web interface, to translate speech to text. 
    '''
    def __init__(self):
        '''
        A constructor that initializes the design layout of the web interface.
        param self: The self reference.
        '''
        ## Initialze the interface and define the input as the microphone
        ## and return the filepath of the audio as a text
        self.app = gr.Interface(
            fn=self.process,
            inputs=gr.Audio(sources=["microphone"], type="filepath"), 
            outputs="text")

    def process(self,filepath):
        '''
        A function that uses the whisper python library to trascribe an audio file.
        param self: The self reference.
        param filepath: directory of saved audio.

        return transcribe: Transcribe audio in a String message type. 
        '''
        model = whisper.load_model("base") #tiny, base, small, medium, large
        result = model.transcribe(filepath, language='en', fp16=False) 
        transcribe = result["text"]

        return "Transcript: " + transcribe 

    def run_app(self):
        '''
        A function that launches the app.
        param self: The self reference. 
        '''
        self.app.launch()   

if __name__ == '__main__':
    ## Instantiate the `SpeechToText` class and run the web interface
    cmd = SpeechToText()
    cmd.run_app()

# openai.api_key = "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"
# audio = open(filepath, "rb")
# transcribe = openai.Audio.transcribe("whisper-1",audio)