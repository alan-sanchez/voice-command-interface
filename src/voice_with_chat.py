#!/usr/bin/env python3

## Import python libraries
import whisper
import os
import rospy
import gradio as gr

from openai import OpenAI
from voice_command_interface.srv import Coordinates 
from std_msgs.msg import String


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
        ## Initiate publisher
        self.task_publisher = rospy.Publisher('task', String, queue_size=10)
       
        ## Pull and set key for openai
        self.model_id = model
        key = os.environ.get("openai_key")
        self.client = OpenAI(api_key=key)

        ## Call and wait for cooridnates service
        rospy.wait_for_service('coordinates')
        self.coordinates = rospy.ServiceProxy('coordinates', Coordinates)
        
        ## Speech-to-text setup
        self.whisper_model = whisper.load_model("base")  # tiny, base, small, medium, large

        ## Gradio interface setup
        self.interface = gr.Interface(
            fn=self.process_input,
            inputs=gr.Audio(sources=["microphone"], type="filepath"),
            outputs="text"
        )

        ## Open the file in read mode
        ## Specify the relative path from the home directory and construct the full path using the user's home directory
        relative_path = 'catkin_ws/src/voice_command_interface/prompts'
        action_list_dir = os.path.join(os.environ['HOME'], relative_path, 'action_list.txt')
        examples_dir = os.path.join(os.environ['HOME'], relative_path, 'examples.txt')

        with open(action_list_dir, 'r') as file:
            ## Read the contents of the file into a string
            self.action_list_prompt = file.read()

        with open(examples_dir, 'r') as file:
            ## Read the contents of the file into a string
            self.examples_prompt = file.read()


    def generate_response(self, voice_cmd, max_length=1000):
        '''
        A function that generates a text response.

        Parameters:
        - self: The self reference.
        - prompt (str): The user's input prompt.
        - max_length (int): The maximum length of the generated response.

        Returns:
        - response (str): The generated response.
        '''
        try:
            answer = self.coordinates(1)
        except rospy.ServiceException as e:
            rospy.logwarn('Service call failed for')

        converter = str(answer)
        objects_and_coordinates = converter.replace("\n","").replace("  ","").replace("\\","")

        prompt = self.action_list_prompt + objects_and_coordinates + self.examples_prompt + voice_cmd
        # print(prompt)
        
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="gpt-3.5-turbo", #"gpt-4"
        )

        self.task_publisher.publish(chat_completion.choices[0].message.content) #"[('cup',[0.67, 0.24, 1.05])]")#
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
    ## Initialize the `relax_arm_control` node
    rospy.init_node('voice_with_chat')

    ## Instantiate the `TaskGenerationAndSpeechToText` class
    combined_class = TaskGenerationAndSpeechToText()

    ## Start the Gradio interface
    combined_class.run_interface()
