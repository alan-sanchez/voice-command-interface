#!/usr/bin/env python3

## Import python libraries
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class FastChatGeneration:
    '''
    A class for generating responses using the FastChat T5-based model.
    '''
    def __init__(self, model_id="lmsys/fastchat-t5-3b-v1.0"):
        '''
        A constructor that initializes the tokenizer and model.

        Parameters:
        - self: The self reference.
        - model_id (str): The Hugging Face model identifier.
        '''
        # The Hugging Face model identifier
        self.model_id = model_id

        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, legacy=False)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    def generate_response(self, prompt, max_length=1000):
        '''
        A function that generates a text response.

        Parameters:
        - prompt (str): The user's input prompt.
        - max_length (int): The maximum length of the generated response.
        '''
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        generation_output = self.model.generate(input_ids=input_ids, max_length=max_length)
        return self.tokenizer.decode(generation_output[0])

if __name__ == "__main__":
    # Instantiate the FastChatGeneration class
    fastchat_generator = FastChatGeneration()

    # Start the interactive loop
    while True:
        # Ask the user for input
        user_input = input("Enter your question (or type 'exit' to end): ")

        # Check if the user wants to exit
        if user_input.lower() == "exit":
            print("Exiting the FastChat generator. Goodbye!")
            break

        # Generate response
        generated_response = fastchat_generator.generate_response(user_input, max_length=1000)

        # Print the generated response
        print("Generated Response:", generated_response)
        print("")
