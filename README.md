# AI-Chatbot

Chatbot using token parsing to direct requests to required model for requested output.
*   Digit Recognition Agent - Determines digits from writing in image:
    *   Tensorflow keras dataset and optimisation layers
    *   Mnsist dataset
    *   Trains model on known digit data
    *   Inputs image into trained model and outputs highest matching result
*   Landmark Detection Agent - Identifies landmark in images:
    *   Google cloud vision model
    *   Inputs image into trained model and outputs result if landmark(s) found
*   Sentiment Analysis Agent - Determines sentiment or basic emotion behind a piece of text:
    *   Google cloud language model
    *   Inputs text into trained model and outputs highest matching result
*   Cat Or Dog Agent - Determines if entity in image is a cat or dog:
    *   Tensorflow keras training suite used to make model
    *   Previously trained model using dataset
    *   Inputs image into trained model and outputs result if found
*   Entity Analysis Agent - Determines the living entities in an image:
    *   Google cloud language model
    *   Inputs image into trained model and outputs list of entites found
