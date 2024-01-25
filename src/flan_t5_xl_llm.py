#!/usr/bin/env python3

'''
Helpful Documentation: 
https://huggingface.co/docs/transformers/v4.36.1/en/main_classes/pipelines#transformers.Text2TextGenerationPipeline
'''

## Import python libraries
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, LlamaForCausalLM, T5Tokenizer, T5ForConditionalGeneration
import torch

class OpenLLAMAChatGeneration:
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
        ## The Hugging Face model identifier/checkpoint: https://huggingface.co/google/flan-t5-xl
        self.model_id = "google/flan-t5-xl"

        # Instantiate tokenizer and model
        # self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, legacy=False)
        # self.model = LLamaForCausalLM.from_pretrained(self.model_id, torch_dtype=torch.float16, device_map="auto")

        ## 
        self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_id, device_map="auto")
        
        ## Open the file in read mode
        with open('/home/alan/catkin_ws/src/voice_command_interface/src/init_info.txt', 'r') as file:
            ## Read the contents of the file into a string
            self.file_contents = file.read()

        # # Now, file_contents contains the contents of the file as a string
        # print(file_contents)


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
        # init_info = str("Here is a list of actions that a robot can perform: clean table, clean mug, clean phone, clean bowl, clean keyboard, clean cup, clean spoon, clean knife, clean plate." +
        #                 # "If a command isn't part of the list, let the user know that you don't have that command in your list."
        #                 "return the order of actions based on this voice command if it's on the provided list; if not, let the user know you can't perform the specific task: " + prompt)
        #                 # "return the order of actions based on this voice command: " + prompt)
        
        ##
        request = self.file_contents + prompt
        
        ##
        input_ids = self.tokenizer(request,truncation=True, return_tensors="pt").input_ids.to("cuda")
        generation_output = self.model.generate(input_ids=input_ids, max_length=max_length)
        response = self.tokenizer.decode(generation_output[0])
        
        return response.lstrip('<pad>').rstrip('</s>').strip()

if __name__ == "__main__":
    ## Instantiate the FastChatGeneration class
    Open_generator = OpenLLAMAChatGeneration()

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
