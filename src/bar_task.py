#!/usr/bin/env python3

import rospy
import sys
import os
import json
import ast
import time
import csv

import sounddevice as sd
import soundfile as sf

from gpt_features import SpeechToText, TextToText, TextToSpeech
from scipy.io.wavfile import write
from openai import OpenAI
from std_msgs.msg import String
from halo import Halo

class BarTask():
    """
    A class that handles audio communication using ROS and OpenAI API.
    """
    def __init__(self):
        """
        Initialize the AudioNode class with ROS subscribers, publishers, and OpenAI API key and client. 
        """
        ## Initialize subscriber
        self.object_map_sub      = rospy.Subscriber('object_map',      String, self.map_update_callback)
        self.cleaning_status_sub = rospy.Subscriber('cleaning_status', String, self.status_callback)
        self.moved_obj_sub       = rospy.Subscriber('moved_object',    String, self.moved_obj_callback)
        
        ## Initialize publisher
        self.known_obj_pub         = rospy.Publisher('known_object_dict',    String, queue_size=10)
        self.unknown_obj_pub       = rospy.Publisher('unknown_object_label', String, queue_size=10)
        self.human_demo_status_pub = rospy.Publisher('human_demo_status',    String, queue_size=10)
        
        ## Specify the relative, prompt, and audio directory paths
        self.relative_path = 'catkin_ws/src/voice_command_interface/'
        self.cocktail_filename_dir = os.path.join(os.environ['HOME'], self.relative_path, 'prompts/cocktail_prompt.txt') 
        self.repo_filename_dir = os.path.join(os.environ['HOME'], self.relative_path, 'prompts/label_prompt.txt')
        self.base_dir = os.path.join(os.environ['HOME'], self.relative_path, 'data') 
        self.transcript_dir = os.path.join(os.environ['HOME'], self.relative_path, 'audio_files/audio_transcripts')
        ## Read the file contents
        with open(self.repo_filename_dir, 'r') as file:
            lines = file.readlines()
        self.repo_list = ast.literal_eval(lines[-1].strip())

        ## Get the OpenAI API key from environment variables
        self.key = os.environ.get("openai_key")
        
        ## Initialize the OpenAI client with the API key
        self.client = OpenAI(api_key=self.key)

        ## Initialize SpeechToText, TextToText, and TextToSpeech classes
        self.stt = SpeechToText()
        self.ttt = TextToText()
        self.tts = TextToSpeech()

        ## Recording parameters: sampling rate and duration
        self.fs = 44100  # Sampling rate in Hz
        self.default_time = 10 # Duration in seconds

        ## Flag for initial object list
        self.ingredient_list = []
        self.object_map_dict = {}
        self.flag = True
        self.temp_filename = 'temp_speech.wav'

        ## Initialze time tracker for saved text
        self.start_time = None
        self.start_flag = True

        ## Create folder to store data
        self.save_dir = self.create_incremented_directory()

        ## Initialize Halo spinner for indicating processing
        self.spinner = Halo(text='Computing response', spinner='dots')

        ## Log initialization notifier
        rospy.loginfo('{}: is ready.'.format(self.__class__.__name__))

        self.pull_transcript('after_demo.txt')
        
    def pull_transcript(self,filename):
        '''
        
        '''
        file_dir = os.path.join(self.transcript_dir,filename)
        ## Read the file contents
        with open(file_dir, 'r') as file:
            fetch_transcript = file.read()
        return fetch_transcript


    def moved_obj_callback(self, str_msg):
        '''
        Callback function to handle a message indicating a moved object.

        Parameters:
        - str_msg (String): The message containing data about the moved object.
        '''
        self.save_info(str_msg.data, 'moved')


    def create_incremented_directory(self):
        '''
        Create a new incremented directory for saving data.

        Returns:
        - incremented_dir (String): The path to the newly created directory.
        '''
        # Check if the base directory exists
        if not os.path.exists(self.base_dir):
            ## Create the base directory where data directories will be stored.
            os.makedirs(self.base_dir)
            
            ## Create the first incremented directory named 'data_0'
            incremented_dir = os.path.join(self.base_dir, "data_0")
            os.makedirs(incremented_dir)
            
            ## Return the path to the newly created directory
            return incremented_dir

        ## Find the highest incremented directory name
        existing_dirs = [d for d in os.listdir(self.base_dir) if os.path.isdir(os.path.join(self.base_dir, d)) and d.startswith("data_")]
        existing_indices = [int(d.split('_')[1]) for d in existing_dirs if d.split('_')[1].isdigit()]

        ## Determine the new directory index by finding the maximum existing index and adding one.
        if existing_indices:
            new_index = max(existing_indices) + 1
        else:
            # If there are no existing directories, start with index 0
            new_index = 0

        ## Create a new directory with the incremented index
        incremented_dir = os.path.join(self.base_dir, f"data_{new_index}")
        os.makedirs(incremented_dir)
        
        ## Return the path to the newly created directory.
        return incremented_dir
    
    def save_info(self, content, process):
        '''
        Save the transcript to a file in the save directory.

        Parameters:
        - content (String): The text content to be saved.
        - process (String): The process that generated the content. It determines which column the content will be saved in.
        '''
        ## Check if the start_time is set, and calculate the elapsed time since start_time
        if self.start_time == None:
            elapsed_time = "N/A"
        else:
            elapsed_time = str(round(time.time() - self.start_time,2))
        
        ## Define the path where the CSV file will be saved
        save_path = os.path.join(self.save_dir, 'transcript.csv')        

        ## Define the header row for the CSV file
        header = ["Timestamp", "Fetch's transcript", "Transcribed audio", "Fetch's response to transcribed audio", "Drink", "Ingredients", "Moved object", "Map"]

        ## Initialize a new row with the elapsed time adn empty string for each content column
        row = [elapsed_time, "", "", "", "", "", "", ""]

        ## Determine which column to place the content based on the process argument
        if process == 'fetch':
            row[1]=content
        elif process == 'whisper':
            row[2] = content
        elif process == 'response':
            row[3] = content.replace('\n', ' ')
        elif process == 'drink':
            row[4] = content
        elif process == 'ingredients':
            row[5] = content
        elif process == "moved":
            row[6] = content
        elif process == 'map':
            row[7] = content

        ## Append the row to the CSV file
        with open(save_path, mode='a', newline='') as file:
            writer = csv.writer(file)

            ## Write the header row if the file is new/empty
            if os.path.getsize(save_path) == 0:
                writer.writerow(header)
            
            ## Write the row with the timestamp and content into the CSV file
            writer.writerow(row)


    def append_text_to_file(self, filename, text):
        '''
        Append text to a file after removing the last line.

        Parameters:
        - filename (String): The path to the file.
        - text (String): The text to append.
        '''
        ## Read the file contents
        with open(filename, 'r') as file:
            lines = file.readlines()

        ## Read the file contents
        lines = lines[:-1]
        lines.append(text)

        ## Write the modified contents back to the file
        with open(filename, 'w') as file:
            file.writelines(lines)


    def status_callback(self, str_msg):
        '''
        Callback function for handling the status of the cleaning robot. If status is complete, notify the user

        Parameters:
        - str_msg (String): A string message indicating the status of the cleaning process.
        '''
        if str_msg.data == "complete":
            self.tts.playback("complete.wav")
            fetch_transcript = self.pull_transcript('complete.txt')
            self.save_info(fetch_transcript, 'fetch')


    def map_update_callback(self, msg):
        '''
        Callback function for the 'object_map' topic. Updates the object map dictionary
        and modifies the system prompt file with the received message.

        Parameters:
        - msg (String): dictionary of object_map in a string format.     
        '''        
        print()
        ## Load the object map dictionary from the received message
        self.save_info(msg.data, 'map')
        self.object_map_dict = json.loads(msg.data)
        self.append_text_to_file(filename=self.cocktail_filename_dir, text= msg.data)
        contaminated_objects = {key: value for key, value in self.object_map_dict.items() if value['status'] != 'clean'} 
        
        # print(contaminated_objects)

        if len(self.ingredient_list) != 0:
            rospy.sleep(2)
            all_objs_used = all(obj in contaminated_objects for obj in self.ingredient_list)
            # print(all_objs_used)
        
            if all_objs_used and self.flag == True:
                # print("made it here")
                message = "Are you finished making your " + self.drink
                self.flag = False
                self.tts.convert_to_speech(text=message, filename=self.temp_filename)
                self.tts.playback(self.temp_filename)
                self.save_info(message, 'fetch')


    def record_audio(self, filename=None):
        """
        Method to record audio, convert it to text, and get a response from OpenAI API.

        Parameters:
        - system_filename (None or String): the filename of the system prompt. If set to None, the
          text_to_text method will set a default system prompt. 
   
        Returns:
        -Response(dictionary): reponse of GPT in a dictionary format. 
        """
        if self.start_flag == True:
            self.start_time = time.time()
            self.tts.playback("hello.wav")
            fetch_transcript = self.pull_transcript('hello.txt')
            self.save_info(fetch_transcript, 'fetch')
            self.start_flag = False
        
        ## Prompt the user to start recording
        input("Press Enter to start recording\n")
        ## Record audio from the microphone
        self.myrecording = sd.rec(int(self.default_time * self.fs), samplerate=self.fs, channels=2)
        input('Recording... Press Enter to stop.\n')  # Wait for the user to press Enter to stop the recording
        sd.stop()
        
        ## Start the spinner to indicate processing
        self.spinner.start()

        ## create temporary file name and save the recorded audio as a WAV file
        temp_filename = 'temp_recording.wav'
        write(temp_filename, self.fs, self.myrecording)  # Save as WAV file 
        
        ## Use whisper speech to text (stt) converter
        transcript = self.stt.convert_to_text(temp_filename)
        # print(transcript)
        self.save_info(transcript, 'whisper')
        os.remove(temp_filename)

        ## Get the response from OpenAI API
        response = self.ttt.text_to_text(system_filename=filename, user_content=transcript)
        response = response.replace("```python ", "").replace("```", "").strip()
        response = response.replace("’","").strip()
        response = response.replace("python ", "").strip()
       
        ## Log the response
        self.save_info(response, 'response')
        
        ## Return the response which is either a dictionary or list
        return ast.literal_eval(response)


    def get_task(self, dict_response):
        '''
        Process the response dictionary and execute tasks based on the keys.

        Parameters:
        - dict_repsonse (dictionary): dictionary returned from gpt.
        '''        
        ## Extract key from the response dictionary (Assuming the dictionary length size is 1)
        key_list = list(dict_response.keys())
        
        ## Filter out clean objects from dictionary
        filtered_dict = {key: value for key, value in self.object_map_dict.items() if value['status'] != 'clean'}         
        
        ## Extract the values (labels) for the contaminated objects
        value_list = list(dict_response.values())

        ############################
        ## drink selection condition
        ############################
        if key_list[0] == 'A':
            self.spinner.stop()
            
            ## Extract the drink information from the dicionary and save it
            self.drink = dict_response['A']['drink']
            self.save_info(self.drink,'drink')

            ## Extract the drink information from the dictionary and save it
            self.ingredient_list= dict_response['A']['ingredients']
            self.save_info(self.ingredient_list, 'ingredients')

            ## Playback an audio file indicating the start of the mixing process
            self.tts.playback('begin_mixing.wav') # Audio was created before the developement of this script
            fetch_transcript = self.pull_transcript('begin_mixing.txt')
            self.save_info(fetch_transcript, 'fetch')

        ##########################
        ## Known objects condition
        ##########################
        elif key_list[0] == 'B':
            self.spinner.stop()
            self.tts.playback('disinfection_notification.wav') # Audio was created before the developement of this script
            fetch_transcript = self.pull_transcript('disinfection_notification.txt')
            self.save_info(fetch_transcript, 'fetch')
            rospy.sleep(.3)
            self.known_obj_pub.publish(str(filtered_dict))
                
        ############################
        ## Unknown objects condition
        ############################
        elif key_list[0] == 'C':
            self.spinner.stop()
            
            ## Pull lables of unknown objects
            unkown_obj_names = [item for sublist in value_list for item in sublist]

            ## Create question for human operator
            message = f"{', '.join(unkown_obj_names)} is not in my repository of known items. Would you like to show me how by guiding my arm?"# Or would you rather I disinfect items that are in my repository?"

            ## Convert message to speech and play back for human operator
            self.tts.convert_to_speech(message, self.temp_filename)
            self.tts.playback(self.temp_filename)
            self.save_info(message, 'fetch')
            self.tts.del_speech_file(self.temp_filename)

            ## Record response of human operator
            response = self.record_audio(filename='new_trajectory.txt')
            self.spinner.stop()
            
            ## Process the response from the human operator
            for key, value in response.items():
                if value == 1:
                    self.human_demo_status_pub.publish('pause')
                    
                    ## Process each unkown object label
                    for label in unkown_obj_names:
                        self.unknown_obj_pub.publish(label)

                        ## Generate and play back message for duiing the arm
                        message = f"Okay, show me how to disifect the {label} "
                        self.tts.convert_to_speech(message, self.temp_filename)
                        self.tts.playback(self.temp_filename)
                        self.save_info(message, 'fetch')

                        ## Publish relaxation status for the arm
                        self.human_demo_status_pub.publish('relax')
                        self.tts.playback('relax_arm.wav')
                        fetch_transcript = self.pull_transcript('relax_arm.txt')
                        self.save_info(fetch_transcript, 'fetch')
                        
                        ## Prompt the user to start and stop recording arm trajectory
                        input("\n Press Enter to start recording arm trajectory")
                        self.tts.playback('recording_path_notification.wav')
                        fetch_transcript = self.pull_transcript('recording_path_notification.txt')
                        self.save_info(fetch_transcript, 'fetch')
                        self.human_demo_status_pub.publish('start')

                        input("\n Press Enter to stop recording the guided path")                        
                        self.tts.playback('saving_traj.wav')
                        fetch_transcript = self.pull_transcript('saving_traj.txt')
                        self.save_info(fetch_transcript, 'fetch')
                        self.human_demo_status_pub.publish('finish')
                        
                        ## Return the arm to the initial pose
                        rospy.sleep(1)
                        self.human_demo_status_pub.publish('init_pose') 
                        rospy.sleep(6)
                        
                        ## Update the labe list with the new object
                        self.repo_list.append(label)
                        self.append_text_to_file(filename=self.repo_filename_dir, text = str(self.repo_list))
                    
                    ## Notify the user after the demonstration
                    # print(filtered_dict)
                    self.tts.playback('after_demo.wav')
                    fetch_transcript = self.pull_transcript('after_demo.txt')
                    self.save_info(fetch_transcript, 'fetch')
                    rospy.sleep(1)
                    self.known_obj_pub.publish(str(filtered_dict))
                    
                else: 
                    ## Filter known items and publish the dictionary
                    known_item_dict = {key: value for key, value in filtered_dict.items() if value['in_repo'] == True}
                    self.known_obj_pub.publish(str(known_item_dict))

        else:
            self.spinner.stop()
            self.tts.playback("no_contamination.wav")
            fetch_transcript = self.pull_transcript('no_contamination.txt')
            self.save_info(fetch_transcript, 'fetch')

        self.flag = True
        # self.tts.del_speech_file(filename=self.temp_filename)    

    def run(self):
        while not rospy.is_shutdown():
            dict_response = self.record_audio('cocktail_prompt.txt')
            # print(dict_response)
            task = self.get_task(dict_response)

        

if __name__ == '__main__':
    ## Initialize the ROS node with the name 'audio_node'
    rospy.init_node('bar_task', argv=sys.argv)

    ## Create an instance of the AudioCommunication class
    obj = BarTask()
    
    ## 
    input("press enter to start operation")

    # while True:
    #     dict_response = obj.record_audio('cocktail_prompt.txt')
    #     # print(dict_response)
    #     task = obj.get_task(dict_response)
    try:
        obj.run()
    except rospy.ROSInterruptException:
        pass
   

