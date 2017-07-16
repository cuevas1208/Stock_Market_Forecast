## File: model_calls.py
## Name: Manuel Cuevas
## Date: 01/14/2017
## Project: CarND - LaneLines
## Desc: convolution neural network model calls to train and use the model
## Revision: Rev 0000.004
#######################################################################################
#importing useful packages
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from sklearn.utils import shuffle
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import glob
import os.path
from model_architecture import *

BATCH_SIZE = 128   #++BATCH_SIZE ++Faster --growth


# Features and Labels
x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
#one_hot_y = tf.one_hot(y, 43) no need for one_hot

weights = tf.Variable(tf.zeros([1, 1], dtype=tf.float32))
bias = tf.Variable(tf.ones([1, 1], dtype=tf.float32))

logits = model(x)

#Evaluation Model
saver = tf.train.Saver()
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


'''
This model uses 3 convolution layers with filter size 7x7, 3x3, and 4x4.
ELU activations, after every convolution layer and Max pooling layer.
After the convolution layers I am using 2 hidden layers with dropout
and a 43 output.
Input:  32x32 image
Return: logits'''
def training_model(X_train, y_train, X_valid, y_valid, rate = 0.001, classes = 43):
    #Training pipeline
    #prediction = tf.nn.softmax(logits)
    #cross_entropy = -tf.reduce_sum(one_hot_y * tf.log(prediction), reduction_indices=1)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
    loss_operation = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate = rate)
    training_operation = optimizer.minimize(loss_operation)
    
    #training model
    import time
    last_accuracy = 0
    EPOCHS = 50

    beginTime = time.time()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        num_examples = len(X_train)

        print("Training...")
        print()
        for i in range(EPOCHS):
            X_train, y_train = shuffle(X_train, y_train)
            for offset in range(0, num_examples, BATCH_SIZE):
                end = offset + BATCH_SIZE
                batch_x, batch_y = X_train[offset:end], y_train[offset:end]
                sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
                
            validation_accuracy = evaluate(X_valid, y_valid)
            print("EPOCH {} ...".format(i+1))
            print("Validation Accuracy = {:.3f}".format(validation_accuracy))
            print()

            #Stop Training
            if (last_accuracy > (validation_accuracy + .002)):
                break
            else:
                last_accuracy = validation_accuracy

        saver.save(sess, 'saved_models/model.ckpt')
        print("Model saved")
    
    endTime = time.time()
    print('Total time: {:5.2f}s'.format(endTime - beginTime))
    return saver

'''
Evaluate the module
Input:  X_data, y_data - Data set(32x32 images), and labels
Return: accuracy of the model compared to the data set '''
def evaluate(X_data, y_data):    
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

'''
Output the prediction label for the input image, and plot the results
Input:  images - array image
        imgLabes - images labels
        save_file - file to upload model
        y_train, X_train - training data and labels
Return: None                                         '''
def predict(images, imgLabes, save_file, y_train, X_train):
    ### Load the images and plot them here.
    fig = []

    saver = tf.train.import_meta_graph(save_file+'.meta')
    
    # Placeholder
    x = tf.placeholder(tf.float32, (None, 32, 32, 1))
    logits = model(x)
    #soft = tf.nn.softmax(logits)
    top_k = tf.nn.top_k(logits,3)

    #Returns the index with the largest value across axes of a tensor.
    pred = tf.argmax(logits , 1)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, save_file)

        for i in range (0,len(images)):
            resized_image = tf.image.resize_images(images[i], [32, 32])
            result = sess.run(resized_image)
            reshaped_image=result.reshape(( 1,32,32,1))
            preditcion = sess.run(pred, feed_dict={x: reshaped_image})
            top_indices = sess.run(top_k, feed_dict={x: reshaped_image})
            print("Image ", i, top_indices.indices)

            #print images
            fig=plt.figure()
            data=np.arange(900).reshape((30,30))
            ax=fig.add_subplot(2,2,2)        
            ax.imshow(images[i])
            ax=fig.add_subplot(2,2,1)
            for ind in range (len(y_train)):
                if (int(preditcion) == y_train[ind]):
                    ax.imshow(X_train[ind])
                    break

            print(" Preditcion Image Label = {}        Web Image Label =".format(preditcion), os.path.splitext(os.path.basename(imgLabes[i]))[0])
            plt.show()

