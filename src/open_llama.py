#!/usr/bin/env python3

##
import torch
from transformers import AutoTokenizer, LlamaForCausalLM

class LLAMAGeneration:
    '''
    A class for generating responses using the OpenLLAMA model: https://github.com/openlm-research/open_llama
    '''
    def __init__(self):
        '''
        A constructor that initializes the toeknizer and model.
        
        Parameters:
        - self: The self reference.
        '''
        ## The hugging face model identifier
        self.model_id = model_id="openlm-research/open_llama_7b_v2"
        
        ## Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, legacy=False)
        self.model = LlamaForCausalLM.from_pretrained(self.model_id,
                                                      torch_dtype=torch.float16,#bfloat16, 
                                                      # offload_folder="offload", 
                                                      device_map="auto"
        )

    def generate_response(self, prompt, max_new_tokens=300):
        '''
        A function that generates a text response

        Parameters:
        - prompt (str): The user's input prompt.
        - max_new_tokens (int): The maximum number of tokens to generate
        '''
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to('cuda')
        generation_output = self.model.generate(input_ids=input_ids, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(generation_output[0])

if __name__ == "__main__":
    ## Instantiate the LLAMAGeneration class
    llama_generator = LLAMAGeneration()

    ## Start the interactive loop
    while True:
        ## Ask the user for input
        user_input = input("Enter your question (or type 'exit' to end): ")

        ## Check if the user wants to exit
        if user_input.lower() == "exit":
            print("Exiting the LLAMA generator. Goodbye!")
            break

        ## Generate response
        generated_response = llama_generator.generate_response(user_input, max_new_tokens=500)

        # Print the generated response
        print("Generated Response:", generated_response[3:])
        print("")
