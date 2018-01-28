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
from data_Prepare import getclassesTotal

class modelClass:
    
    def __init__(self):
        #self.model = Sequential()
        self.get_model()
        
    
    def get_model(self):
        self.model = Sequential()
        # Preprocess incoming data
        # Normalize the features using Min-Max scaling centered around zero and reshape
        self.model.add(Lambda(lambda x: (x/6) - 1., input_shape=(32, 32, 1), output_shape=(32, 32, 1)))
        print(self.model.output_shape)
        #self.model.add(Dense(32, input_shape=(32, 32, 1)))
        self.model.add(Convolution2D(4, 2, 2))
        print(self.model.output_shape)
        self.model.add(AveragePooling2D((2, 2)))
        self.model.add(Activation('relu'))
        print(self.model.output_shape)

        self.model.add(Convolution2D(8, 1, 1))
        self.model.add(Activation('relu'))
        print(self.model.output_shape)
        
        self.model.add(Convolution2D(16, 2, 2))
        print(self.model.output_shape)
        self.model.add(AveragePooling2D((2, 2)))
        self.model.add(Activation('relu'))
        print(self.model.output_shape)
        
        self.model.add(Flatten())
        self.model.add(Activation('relu'))
        print(self.model.output_shape)
        
        self.model.add(Dense(200))
        self.model.add(Activation('relu'))
        classes = getclassesTotal()
        self.model.add(Dense(classes))
        if (classes > 1):
            model.add(Dense(classes, activation='softmax'))
            self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        else:
            self.model.compile(optimizer="adam", loss="mse", metrics=['accuracy'])
        print(self.model.output_shape)
        

    '''
    Stores the trained model in json format
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





