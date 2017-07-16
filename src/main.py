## File: main.py
## Name: Manuel Cuevas
## Date: 2/08/2016
## Project: CarND - Behavioral Cloning 
## Desc: Load, Preprocces, trains and saves car simulation images.
## Usage: This project identifies a series of steps that identify stering wheel
## angle from images.
## Tools learned in Udacity CarND program were used to identify lane lines on the road.
#######################################################################################
#importing useful packages
import pickle
import math
import numpy as np
import keras


from keras_model import modelClass
from data_Load import *
from data_Prepare import *

def genImage(limit, batch, x_data, y_data):
    while 1:
        for i in range(0, limit, batch):
            x = x_data[i : i + batch]
            y = y_data[i : i + batch]
            yield [np.array(x), np.array(y)]


if __name__ == '__main__':
    #load data
    dataDates = []
    dataDates.append('2010-01-01')
    dataDates.append(datetime.now().date())

    x_data, y_data = get_detaSet(dataDates, 'TSLA')

    ##########################################################################################
    # One_hot encoder and split data
    ##########################################################################################  
    x_data, y_data= roundDataSet(x_data, y_data)
    #y_data = oneHot(y_data)
    x_train, x_test, y_train, y_test, x_valid, y_valid = splitData(x_data, y_data)

    ##########################################################################################
    # model setup
    ########################################################################################## 
    #Get model
    keras_model = modelClass()

    #Prepare data batch
    batchSize = 32
    sampPerEpoch = len(x_train)-(len(x_train)%batchSize)
    valSamp =      len(x_test) -(len(x_test)%batchSize)
    epochs = 10

    ##########################################################################################
    # running the model
    ##########################################################################################
    #train the model using the generator function
    estimator = keras_model.model.fit_generator(genImage(sampPerEpoch, batchSize, x_train, y_train),
                               validation_data = genImage(valSamp, batchSize, x_test, y_test),
                               nb_val_samples=valSamp, samples_per_epoch = sampPerEpoch,
                               nb_epoch=epochs, verbose = 1)##Saving the model

    
    predictions = estimator.predict(x_valid)
    print(predictions)
    #print(encoder.inverse_transform(predictions))
    #Save model
    keras_model.model.savemodel(location = "./TSLA_model")

