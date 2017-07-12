## File: model.py
## Name: Manuel Cuevas
## Date: 02/14/2017
## Project: CarND - Behavioral Cloning
## Desc: convolution neural network model to learn a track
## Ref: https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
#######################################################################################
'''
This model was based on the Nvidia model with modification to better fit our needs
This model uses 3 convolutional layers with filter size 7x7, 1x1, 3X3 followed by a
elu activations.
Input:  image size -> 160, 320, 3
Return: logits'''
from keras.layers.convolutional import Convolution2D, Cropping2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Reshape, Dropout
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers import Lambda, ELU

class modelClass:
    
    def __init__(self):
        self.model = Sequential()
        self.get_model()
        print("hey")
        
    
    def get_model(self):
        # Preprocess incoming data
        # Normalize the features using Min-Max scaling centered around zero and reshape
        self.model.add(Convolution2D(32, 3, 1, input_shape=(32, 32, 1)))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Dropout(0.5))
        self.model.add(Activation('relu'))
        self.model.add(Flatten())
        self.model.add(Dense(128))
        self.model.add(Activation('relu'))
        self.model.add(Dense(1))
        # compile self.model using adam
        self.model.compile(optimizer="adam", loss="mse")

    '''
    Stores the trained model to json format
    Input:  model - trained model
            location, - folder location
    Return: void '''
    def savemodel(self, location = "./steering_model"):
        import os
        import json
        print("Saving model weights and configuration file.")

        if not os.path.exists(location):
            os.makedirs(location)

        # serialize model to JSON and weights to h5
        self.model.save_weights(location + "/model.h5", True)
        with open(location +'/model.json', 'w') as outfile:
            outfile.write(self.model.to_json())
            
        print("Saved model to disk")





