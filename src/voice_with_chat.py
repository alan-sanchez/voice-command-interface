#!/usr/bin/env python3

## Import python libraries
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import gradio as gr
import whisper
import openai

class TaskGenerationAndSpeechToText:
    '''
    A class that combines task generation and speech-to-text functionality.
    '''
    def __init__(self):
        '''
        A constructor that initializes the tokenizer, model, and Gradio interface.

        Parameters:
        - self: The self reference.
        '''
        ## Task generation setup
        self.fastchat_model_id = "lmsys/fastchat-t5-3b-v1.0" # Provide the model id/checkpoint
        self.fastchat_tokenizer = AutoTokenizer.from_pretrained(self.fastchat_model_id, legacy=False)
        self.fastchat_model = AutoModelForSeq2SeqLM.from_pretrained(self.fastchat_model_id, torch_dtype=torch.float16, device_map="auto")

        ## Speech-to-text setup
        self.whisper_model = whisper.load_model("base")  # tiny, base, small, medium, large

        ## Gradio interface setup
        self.interface = gr.Interface(
            fn=self.process_input,
            inputs=gr.Audio(sources=["microphone"], type="filepath"),
            outputs="text"
        )

    def generate_response(self, prompt, max_length=1000):
        '''
        A function that generates a text response based on the provided prompt.

        Parameters:
        - self: The self referentce. 
        - prompt (str): The user's input prompt.
        - max_length (int): The maximum length of the generated response.

        Returns:
        - response (str): The generated response.
        '''
        ## Create initial information for the voice command
        init_info = f"Here is a list of actions that a robot can perform: clean table, clean mug, clean phone, clean bowl, clean keyboard, clean cup, clean spoon, clean knife, clean plate. Return the order of actions based on this voice command: {prompt}"
        
        ## Create the input ids from tokenizer
        input_ids = self.fastchat_tokenizer(init_info, return_tensors="pt").input_ids.to("cuda")
        generation_output = self.fastchat_model.generate(input_ids=input_ids, max_length=max_length)
        response = self.fastchat_tokenizer.decode(generation_output[0])
        return response.lstrip('<pad>').rstrip('</s>').strip()

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
