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
from loadData import *

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

    x_train, x_test, y_train, y_test, x_valid, y_valid, classesTotal = get_detaSet(dataDates, 'TSLA')

    #Get model
    keras_model = modelClass()

    ##########################################################################################
    # from my project 
    ##########################################################################################
    #Prepare data batch
    batchSize = 120
    sampPerEpoch = len(x_train)-(len(x_train)%batchSize)
    valSamp =      len(x_test) -(len(x_test)%batchSize)
    epoch = 5

##    from sklearn.preprocessing import LabelBinarizer
##    label_binarizer = LabelBinarizer()
##    y_one_hot = label_binarizer.fit_transform(y_train)

    #keras_model.model.compile('adam', 'categorical_crossentropy', ['accuracy'])
    #train the model using the generator function
    keras_model.model.fit_generator(genImage(sampPerEpoch, batchSize, x_train, y_train),
                               validation_data = genImage(valSamp, batchSize, x_test, y_test),
                               nb_val_samples=valSamp, samples_per_epoch = sampPerEpoch,
                               nb_epoch=5, verbose = 1)##Saving the model

    #Save model
    keras_model.model.savemodel(location = "./steering_model")

##    ##########################################################################################
##    # from traffic signs
##    ##########################################################################################
##    x_train, x_test, y_train, y_test, x_valid, y_valid = np.array(x_train),np.array(x_test), \
##                      np.array(y_train), np.array(y_test), np.array(x_valid), np.array(y_valid)
##    from sklearn.preprocessing import LabelBinarizer
##    label_binarizer = LabelBinarizer()
##    y_one_hot = label_binarizer.fit_transform(y_train)
##
##    keras_model.model.compile('adam', 'categorical_crossentropy', ['accuracy'])
##    history = keras_model.model.fit(x_train, y_one_hot, batch_size=sampPerEpoch, nb_epoch=10, validation_split=0.2)
##
##    y_one_hot_test = label_binarizer.fit_transform(y_test)
##    metrics = keras_model.model.evaluate(x_test, y_one_hot_test)
##    for metric_i in range(len(keras_model.model.metrics_names)):
##        metric_name = keras_model.model.metrics_names[metric_i]
##        metric_value = metrics[metric_i]
##        print('{}: {}'.format(metric_name, metric_value))
##
##    ##########################################################################################
##    # here's a more "manual" example
##    ##########################################################################################
##    x_train, x_test, y_train, y_test, x_valid, y_valid = np.array(x_train),np.array(x_test), \
##                      np.array(y_train), np.array(y_test), np.array(x_valid), np.array(y_valid)
##    ##Examples from the keras
##    from keras.preprocessing.image import ImageDataGenerator
##    datagen = ImageDataGenerator(
##        featurewise_center=True,
##        featurewise_std_normalization=True,
##        rotation_range=20,
##        width_shift_range=0.2,
##        height_shift_range=0.2,
##        horizontal_flip=True)
##
##    # compute quantities required for featurewise normalization
##    # (std, mean, and principal components if ZCA whitening is applied)
##    datagen.fit(x_train)
##
##    # fits the model on batches with real-time data augmentation:
##    keras_model.model.fit_generator(datagen.flow(x_train, y_train, batch_size= batchSize),
##                        samples_per_epoch = len(x_train) / sampPerEpoch,
##                        nb_epoch = epoch, verbose = 1)##Saving the model
##
##    ##########################################################################################
##    # here's a more "manual" example
##    ##########################################################################################
##    for e in range(epoch):
##        print('Epoch', e)
##        batches = 0
##        for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size= batchSize):
##            keras_model.model.fit(x_batch, y_batch)
##            batches += 1
##            if batches >= len(x_train) / batchSize:
##                # we need to break the loop by hand because
##                # the generator loops indefinitely
##                break



