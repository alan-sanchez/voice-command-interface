#!/usr/bin/env python3

## Import python libraries
import torch
import os
from openai import OpenAI

class TextGeneration:
    '''
    A class for generating responses using the FastChat T5-based model.
    '''
    def __init__(self, model = "gpt-4"):
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
            model=self.model_id#"gpt-4"
        )

        return chat_completion.choices[0].message.content

if __name__ == "__main__":
    ## Instantiate the FastChatGeneration class
    Open_generator = TextGeneration()

    ## Start the interactive loop
    while True:
        ## Ask the user for input
        user_input = input("What would you like Fetch to disinfect? (or type 'exit' to end): ")

        ## Check if the user wants to exit
        if user_input.lower() == "exit":
            print("Exiting the FastChat generator. Goodbye!")
            break

        ## Generate response
        generated_response = Open_generator.generate_response(user_input, max_length=1000)

        ## Print the generated response
        print("Generated Response:", generated_response)
        print("")
