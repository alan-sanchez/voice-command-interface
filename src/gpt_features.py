#!/usr/bin/env python3
import os
import cv2
import base64
import requests
import sounddevice as sd
import soundfile as sf
import signal
from openai import OpenAI
# from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# Signal handling for interruption
interrupted = False

def signal_handler(signal, frame):
    global interrupted
    interrupted = True

signal.signal(signal.SIGINT, signal_handler)


class OpenAIBase:
    def __init__(self):
        """
        Constructor method for initializing the `OpenAIBase` class.
        """
        ## Pull openai_key, create a client, and set the relative path
        self.key = os.environ.get("openai_key")
        self.client = OpenAI(api_key=self.key)
        self.relative_path = 'catkin_ws/src/voice_command_interface/'



class SpeechToText(OpenAIBase):
    '''
    A class that handles text to speech conversion and audio playback using OpenAI API.
    '''
    def __init__(self):
        '''
        Constructor method for initialzing inherited class.
        '''
        super().__init__()

    def convert_to_text(self, filename="intro"):
        '''
        Converts text to speech and saves it to a file.
        Link:https://platform.openai.com/docs/guides/speech-to-text
        
        Parameters:
        text (str): The text to convert to speech.
        filename (str): The filename to save the speech audio.

        Return:
        transcription.txt (str): The transcribed audio.
        '''
        ## Define audio directory
        if filename == "intro":
            filename = os.path.join(os.environ['HOME'], self.relative_path, 'audio_files', filename + '.wav') 

        ## Ready file
        with open(filename, "rb") as audio_file:
            transcription = self.client.audio.transcriptions.create(model="whisper-1", file=audio_file)
        
        ## Return transcibed text
        return transcription.text



class TextToSpeech(OpenAIBase):
    """
    A class that handles text to speech conversion and audio playback using OpenAI API.
    """
    def __init__(self):
        '''
        Constructor method for initializing inherited class.
        '''
        super().__init__()

    def convert_to_speech(self, text='Hello World!', filename="speech.wav"):
        """
        Converts text to speech and saves it to a file.
        Link:https://platform.openai.com/docs/guides/text-to-speech
        
        Parameters:
        text (str): The text to convert to speech.
        filename (str): The filename to save the speech audio.
        """
        ## Create a speech response using OpenAI's API
        response = self.client.audio.speech.create(model="tts-1", voice="alloy", input=text)

        ## Construct the full file path for saving the audio file and save the response
        file_dir = os.path.join(os.environ['HOME'], self.relative_path, 'audio_files', filename)
        response.stream_to_file(file_dir)
        # print(filename + " saved")

    def playback(self, filename):
        """
        Plays back the specified audio file if it exists.
        
        Parameters:
        filename (str): The filename of the audio file to play back.
        """
        ## Construct the full file path for the audio file
        file_dir = os.path.join(os.environ['HOME'], self.relative_path, 'audio_files', filename )
        
        ## Check if the file exists, if so, then play it back
        if os.path.exists(file_dir):
            data, fs = sf.read(file_dir, dtype='float32')
            sd.play(data, fs)
            # print(f"Playing {file_dir}")
            status = sd.wait()
        else:
            print(f"File {file_dir} does not exist.")



class VisionToText(OpenAIBase):
    '''
    A class that combines task generation and speech-to-text functionality.
    '''
    def __init__(self):
        '''
        Constructor method for initializing inherited class and a default image.
        '''
        super().__init__()

        ## Create a `CVBridge` object
        self.bridge = CvBridge()

        ## Set default_image
        default_image_dir = os.path.join(os.environ['HOME'], self.relative_path, 'images/vlm.jpeg')
        image_bgr = cv2.imread(default_image_dir)
        self.default_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    def viz_to_text(self, img='default', bbox=[0, 0, 640, 480], prompt_filename=None, prompt="what do you see?", max_length=1000):
        '''
        A function that performs vision-to-text conversion using OpenAI's API.
        Reference: https://platform.openai.com/docs/guides/vision

        Parameters:
        - img (Image or str): The image to analyze, either as an image object or a string for the default image.
        - prompt (str): The prompt/question to provide context for the image analysis.
        - bbox (list): The bounding box coordinates [x_min, y_min, x_max, y_max] to crop the image.
        - max_length (int): The maximum number of tokens for the response.
        '''
        ## Use the default image if 'img' is provided as a string
        if isinstance(img, str):
            img = self.default_image
        
        ## Use conditional statement to pull text from the prompt directory
        if prompt_filename != None:
            prompt_dir = os.path.join(os.environ['HOME'], self.relative_path, 'prompts/', prompt_filename)
            with open(prompt_dir, 'r') as file:
                prompt = file.read()
    

        ## Crop the image using the provided bounding box coordinates. Default is the whole image, assuming its size is 640x480 pixels
        cropped_image = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        
        ## Define the temporary image file name, path, and save the cropped image to the the temp directory
        img_name = 'temp.jpeg'
        temp_directory = os.path.join(os.environ['HOME'], self.relative_path, 'images', img_name)
        cv2.imwrite(temp_directory, cropped_image)

        ## Open the saved image file and encode it in base64 format
        with open(temp_directory, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

            ## Set up the headers for the API request
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.key}"
            }

            payload = {
                "model": "gpt-4o",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            },
                        ]
                    }
                ],
                "max_tokens": max_length
            }

            ## Send the POST request to OpenAI's API and retrieve the response and extract the content (text)
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            
        ## Remove the temporary image file
        os.remove(temp_directory)

        ## Return the extracted content
        return content
    

class TextToText(OpenAIBase):
    """
    A class that handles text to speech conversion and audio playback using OpenAI API.
    """
    def __init__(self):
        '''
        Constructor method for initializing inherited class.
        '''
        super().__init__()
        

    def text_to_text(self, system_filename=None, user_prompt='Hello!'):
        '''
        Generates a response from the OpenAI API based on a system prompt and a user prompt.
        Link: https://platform.openai.com/docs/guides/text-generation/chat-completions-api

        Parameters:
        - system_filename (str or None): The filename of the system prompt text file (without extension). Defaults to None.
        - user_prompt (str): The prompt/question provided by the user. Defaults to 'Hello!'.

        Returns:
        - response_content (str): The generated response from the OpenAI API.
        '''
        ## Extract the file path for the system prompt
        if system_filename == None:
            ## Use the default system prompt file if no filename is provided
            system_dir = os.path.join(os.environ['HOME'], self.relative_path, 'prompts', "system_prompt.txt")
        else:
            system_dir = os.path.join(os.environ['HOME'], self.relative_path, 'prompts', system_filename)

        ## Read the system prompt from the specified file
        with open(system_dir, 'r') as file:
            system_prompt = file.read()
                
        ## Create the chat completion request using the OpenAI API
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            )
        
        ## Extract and return the generated response content
        return response.choices[0].message.content

if __name__ == "__main__":
    stt = SpeechToText()
    tts = TextToSpeech()
    vtt = VisionToText()
    ttt = TextToText()


    while True:
        print("\n\nEnter 1 for Speech to Text conversion.")
        print("Enter 2 for Text to Speech conversion and playback.")
        print("Enter 3 for Vision to Text conversion.")
        print("Enter 4 for Text to Text generation.")
        print("Enter 5 to quit\n")
        control_selection = input("Choose an option: ")

        if control_selection == "1":
            text = stt.convert_to_text("intro")
            print(f"Converted Text: {text}")

        elif control_selection == "2":
            tts.convert_to_speech("Hello, Skinny Human.", "intro.wav")
            tts.playback("intro")

        elif control_selection == "3":
            response_text = vtt.viz_to_text()
            print(response_text)

        elif control_selection == "4":
            response_text = ttt.text_to_text()
            print(response_text)
            
        elif control_selection =="5":    
            break

        else:
            print("\nInvalid selection\n")