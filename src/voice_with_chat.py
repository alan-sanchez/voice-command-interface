#!/usr/bin/env python3

## Import python libraries
import torch
import whisper
import os
import gradio as gr
from openai import OpenAI

class TaskGenerationAndSpeechToText:
    '''
    A class that combines task generation and speech-to-text functionality.
    '''
    def __init__(self, model = "gpt-3.5-turbo"):
        '''
         A constructor that initializes the tokenizer and model.

        Parameters:
        - self: The self reference.
        - model_id (str): large language model identifier.
        '''
        ## Pull and set key for openai
        self.model_id = model
        key = os.environ.get("openai_key")
        self.client = OpenAI(api_key=key)

        ## Speech-to-text setup
        self.whisper_model = whisper.load_model("base")  # tiny, base, small, medium, large

        ## Gradio interface setup
        self.interface = gr.Interface(
            fn=self.process_input,
            inputs=gr.Audio(sources=["microphone"], type="filepath"),
            outputs="text"
        )

        ## Open the file in read mode
        with open('/home/alan/catkin_ws/src/voice_command_interface/src/init_info.txt', 'r') as file:
            ## Read the contents of the file into a string
            self.file_contents = file.read()

    def generate_response(self, prompt, max_length=1000):
        '''
        A function that generates a text response.

        Parameters:
        - self: The self reference.
        - prompt (str): The user's input prompt.
        - max_length (int): The maximum length of the generated response.

        Returns:
        - response (str): The generated response.
        '''
        text_content = self.file_contents + prompt
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": text_content,
                }
            ],
            model="gpt-3.5-turbo", #"gpt-4"
        )

        return chat_completion.choices[0].message.content


    def process_input(self, filepath):
        '''
        A function that processes audio input, transcribes it, and generates a response.

        Parameters:
        - self: The self reference. 
        - filepath (str): The directory of the saved audio.

        Returns:
        - result (str): The combined result of transcription and task generation.
        '''
        ## Speech-to-text function
        result = self.whisper_model.transcribe(filepath, language='en', fp16=True)
        transcribe = result["text"]

        ## Task generation based on transcribed text
        generated_response = self.generate_response(transcribe, max_length=1000)

        ## Print the Transcript and LLM response in gradio
        return f"Transcript: {transcribe}\nGenerated Response: {generated_response}"

    def run_interface(self):
        '''
        A function that launches the Gradio interface.

        Parameters:
        - self: The self reference. 
        '''
        self.interface.launch()

if __name__ == "__main__":
    ## Instantiate the `TaskGenerationAndSpeechToText` class
    combined_class = TaskGenerationAndSpeechToText()

    ## Start the Gradio interface
    combined_class.run_interface()
