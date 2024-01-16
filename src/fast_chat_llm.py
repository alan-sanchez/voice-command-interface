#!/usr/bin/env python3

'''
Helpful Documentation: 
https://huggingface.co/docs/transformers/v4.36.1/en/main_classes/pipelines#transformers.Text2TextGenerationPipeline
'''

## Import python libraries
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class FastChatGeneration:
    '''
    A class for generating responses using the FastChat T5-based model.
    '''
    def __init__(self):
        '''
        A constructor that initializes the tokenizer and model.

        Parameters:
        - self: The self reference.
        - model_id (str): The Hugging Face model identifier.
        '''
        # The Hugging Face model identifier/checkpoint
        self.model_id = "lmsys/fastchat-t5-3b-v1.0"

        # Instantiate tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, legacy=False)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id, torch_dtype=torch.float16, device_map="auto")

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
        init_info = str("Here is a list of actions that a robot can perform: clean table, clean mug, clean phone, clean bowl, clean keyboard, clean cup, clean spoon, clean knife, clean plate." +
                        "return the order of actions based on this voice command: " + prompt)
        input_ids = self.tokenizer(init_info, return_tensors="pt").input_ids.to("cuda")
        generation_output = self.model.generate(input_ids=input_ids, max_length=max_length)
        response = self.tokenizer.decode(generation_output[0])
        return response.lstrip('<pad>').rstrip('</s>').strip()

if __name__ == "__main__":
    ## Instantiate the FastChatGeneration class
    fastchat_generator = FastChatGeneration()

    ## Start the interactive loop
    while True:
        ## Ask the user for input
        user_input = input("What would you like Fetch to disinfect? (or type 'exit' to end): ")

        ## Check if the user wants to exit
        if user_input.lower() == "exit":
            print("Exiting the FastChat generator. Goodbye!")
            break

        ## Generate response
        generated_response = fastchat_generator.generate_response(user_input, max_length=1000)

        ## Print the generated response
        print("Generated Response:", generated_response)
        print("")

