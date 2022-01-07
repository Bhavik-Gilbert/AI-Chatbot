#@author Bhavik Gilbert
#Hosts a chatbot which gets responses based on inputs given
#Functionality ->
# Image HandWritten Digit Recognition
# Image Landmark Detection
# Image Cat or Dog Analysis
# Text Sentence Sentiment Analysis
# Text Entity Recognition Analysis
# Text Response


#checking for installation of required google libraries
try:
  !pip install --upgrade google-cloud-vision==2.4.1 
  !pip install --upgrade google-cloud-language
except:
  pass


#importing required libraries
#data manipulation
import numpy as np
#data models
import tensorflow.keras
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.datasets import mnist
#image manipulation
from PIL import Image, ImageOps
import io
#image analysis
from google.cloud import vision
#text analysis
from google.cloud import language
from google.cloud.language import types, enums


#reads txt file and collects responses
#makeshift map
def read_dictionary(file_path):
    responses = {}
    try:
        response_file = open(file_path, "r")
        lines = response_file.readlines()
        response_file.close()
        for line in lines:
            key, value = line.split(":")
            responses[key] = value
        return responses
    except:
        print("Error: Make sure your file is called", file_path)


#unused in code
#creates a txt file with keys and values in the format understood by the chatbot
def create_dictionary():
    file_path = input("Enter a file path (and extension) to save responses to: ")
    response_file = open(file_path, "w")
    while True:
        question = input("Enter a question: ")
        response = input("Enter a response: ")
        response_file.write("{}:{}\n".format(question, response))
        result = input("Would you like to continue (Y/N): ").lower()
        if result != "y":
            break
    response_file.close()

#general agent class, used as a layout
# layout for can_handle which determines if the agent can handle the request 
# layout for get_response which determines how the agent responds
class Agent:
    def can_handle(self,  text):
        pass

    def get_response(self, text):
        pass

#the chatbot class that holds all the agents
#inputs user input into agents and returns agent response to user
#get_response => agent response
class ChatBot:
    def __init__(self):
        self.agents = []

    def add_agent(self, agent):
        self.agents.append(agent)

    def get_response(self, text):
        for agent in self.agents:
            if agent.can_handle(text):
                return agent.get_response(text)

        return "I'm sorry I don't understand " + text

#responds to user based off of key terms in dictionary
#if text in dictionary, response given
#can_handle => returns true if agent can handle text
#get_response => returns dictionary value at key
class ConditionalAgent(Agent):
    def __init__(self, file_path):
        self.conditional_responses = read_dictionary(file_path)
    
    def can_handle(self, text):
        try:
          result = self.conditional_responses[text.lower()]
          return True
        except:
          return False
    
    def get_response(self, text):
        return self.conditional_responses[text.lower()]

#trains digit model and returns number analysed in images
#can_handle => returns true if agent can handle text
#get_response => returns number analysed
class DigitRecognitionAgent(Agent):
    def __init__(self, file_path, image_folder):
        self.conditional_responses = read_dictionary(file_path)
        self.image_path = image_folder

        #training a portion of the model
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(60000, 784)
        x_test = x_test.reshape(10000, 784)
        x_train = x_train.astype("float32")
        x_test = x_test.astype("float32")

        x_train /= 255
        x_test /= 255

        #testing a portion of the model
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

        #optimising the model
        self.model = Sequential()
        self.model.add(Dense(64, input_shape=(784,)))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(10))
        self.model.add(Activation('softmax'))
        self.model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        self.model.fit(x_train, y_train, epochs=4)
        #test results
        score = self.model.evaluate(x_test, y_test)

    def can_handle(self, text):
        try:
          result = self.conditional_responses[text]
          return True
        except:
          return False

    #returns number analysed if image size can be reshaped
    def get_response(self, text):
          try:
            file_path = input("File Path: ")
            file_path = self.image_path + file_path
            #making image into an array
            test_image = np.invert(Image.open(file_path).convert("L"))
            test_image = test_image.reshape(1, 784)
            test_image = test_image.astype("float32")
            test_image /= 255
            pred = self.model.predict(test_image)
            return pred.argmax()
          except:
              return "Invalid file size"

#loads model and returns landmarks detected
#can_handle => returns true if agent can handle text
#get_response => returns landmarks detected
class LandmarkDetectionAgent(Agent):
    def __init__(self, file_path, image_folder):
        self.conditional_responses = read_dictionary(file_path)
        self.image_path = image_folder
        #loading model
        self.client = vision.ImageAnnotatorClient.from_service_account_json("Models/creds.json")
    
    def can_handle(self, text):
        try:
          result = self.conditional_responses[text]
          return True
        except:
          return False
    
    def get_response(self, text):
            file_path = input("File Path: ")
            file_path = self.image_path + file_path
            try:
              with io.open(file_path, "rb") as image_file:
                content = image_file.read()
              #uses vision library to analyse image
              image = vision.Image(content=content)
              response = self.client.landmark_detection(image=image)
              landmarks = response.landmark_annotations
              if len(landmarks) == 0:
                message = "No landmarks found"
              else:
                message = ""
                for landmark in landmarks:
                  message += landmark.description + "\n"
              return message
            except:
              return "Invalid file"

#loads model and returns text sentiment score
#can_handle => returns true if agent can handle text
#get_response => returns text sentiment score
class SentimentAnalysisAgent(Agent):
    def __init__(self, file_path):
        self.conditional_responses = read_dictionary(file_path)
        #loading model
        self.client = language.LanguageServiceClient.from_service_account_json("Models/creds.json")
    
    def can_handle(self, text):
        try:
          result = self.conditional_responses[text]
          return True
        except:
          return False
    
    def get_response(self, text):
            user_input = input("Sentence: ")
            document = types.Document(content=user_input, type=enums.Document.Type.PLAIN_TEXT)
            #using language library to analyse text
            response = self.client.analyze_sentiment(document=document, encoding_type="UTF32")
            sentences = response.sentences
            message = ""
            for sentence in sentences:
              message += sentence.text.content + "\n"
              message += sentence.sentiment.score + "\n"
            return message


#loads model and returns analysis
#can_handle => returns true if agent can handle text
#get_response => returns cat or dog
class CatOrDogAgent(Agent):
    def __init__(self, file_path, model_path, image_folder):
        self.conditional_responses = read_dictionary(file_path)
        self.image_path = image_folder
        np.set_printoptions(suppress=True)
        #loads premade model
        self.model = tensorflow.keras.models.load_model(model_path)
    
    def can_handle(self, text):
        try:
          result = self.conditional_responses[text]
          return True
        except:
          return False
    
    def get_response(self, text):
            file_path = input("File Path: ")
            file_path = self.image_path + file_path
            #makes image into an array and determines input/output size
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            image = Image.open(file_path)
            size = (224, 224)
            image = ImageOps.fit(image, size, Image.ANTIALIAS)
            image_array = np.asarray(image)
            normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
            data[0] = normalized_image_array
            pred = self.model.predict(data)

            #determines outputs
            if(pred.argmax()==1):
              return "I think it's a dog"
            elif(pred.argmax()==0):
              return "I think it's a cat"
            else:
              return "I'm not too sure what it is"

#loads language model and returns information about text
#can_handle => returns true if agent can handle text
#get_response => returns information about text
class EntityAnalysisAgent(Agent):
    def __init__(self, file_path):
        self.conditional_responses = read_dictionary(file_path)
        #loading language module
        self.client = language.LanguageServiceClient.from_service_account_json("Models/creds.json")
    
    def can_handle(self, text):
        try:
          result = self.conditional_responses[text]
          return True
        except:
          return False
    
    def get_response(self, text):
            user_input = input("Sentence: ")
            document = types.Document(content=user_input, type=enums.Document.Type.PLAIN_TEXT)
            #uses language module to analyse text
            response = client.analyze_entities(document=document, encoding_type="UTF32")
            entities = response.entities
            message = ""
            for entity in entities:
              message += entity.name + "\n"
              message += enums.Entity.Type(entity.type).name + "\n"
              message += entity
            return message

#creates chatbot
chatbot = ChatBot()

#sets folder paths
response_folder = "Responses/"
image_folder = "Images/"
model_folder = "Models/"

#creates agent
simple_agent = ConditionalAgent(response_folder + "ConditionalAgent.txt")
digit_agent = DigitRecognitionAgent(response_folder + "DigitAgent.txt", image_folder)
landmark_agent = LandmarkDetectionAgent(response_folder + "LandmarkAgent.txt", image_folder)
sentiment_agent = SentimentAnalysisAgent(response_folder + "SentimentAgent.txt")
cat_or_dog_agent = CatOrDogAgent(response_folder + "CatOrDogAgent.txt", model_folder + "keras_model.h5", image_folder)
entity_agent = EntityAnalysisAgent(response_folder + "EntityAgent.txt")

#adds agents to chatbot
chatbot.add_agent(simple_agent)
chatbot.add_agent(digit_agent)
chatbot.add_agent(landmark_agent)
chatbot.add_agent(sentiment_agent)
chatbot.add_agent(cat_or_dog_agent)
chatbot.add_agent(entity_agent)

#creates arrray of bot names and response folder
bots = [["Digit Recognition Agent" , response_folder + "DigitAgent.txt"], ["Landmark Detection Agent", response_folder + "LandmarkAgent.txt"], 
        ["Sentiment Analysis Agent", response_folder + "SentimentAgent.txt"], ["Cat Or Dog Agent", response_folder + "CatOrDogAgent.txt"],
        ["Entity Analysis Agent", response_folder + "EntityAgent.txt"]]

while True:
    print("Hi, what would you like me to do?")
    user_input = input("")
    #exit code
    if user_input.lower() == "goodbye" or user_input.lower() == "quit":
        break
    #outputs list of commands each bot accepts
    if user_input.lower() == "commands":
      for agents in bots:
        print(agents[0] + "'s trigger words =>")
        commands = read_dictionary(agents[1])
        for command in commands:
          print(command)
        print("")
      continue

    response = chatbot.get_response(user_input)
    print(response)
    print("")