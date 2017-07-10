## File: model_architecture.py
## Name: Manuel Cuevas
## Date: 01/14/2017
## Project: CarND - LaneLines
## Desc: convolution neural network model
## I have adapted the model I used for trafic sign to predict stocks
## https://github.com/cuevas1208/Traffic_Sign_Classifier/blob/production/model_architecture.py
## Revision: Rev 0000.004
#######################################################################################
#importing useful packages
import tensorflow as tf
from tensorflow.contrib.layers import flatten

'''
This model uses 4 convolution layers with filter size 7x7, 3x3, 1X1, and 4x4.
ELU activations, after every convolution layer and Max pooling layer.
After the convolution layers I am using 2 hidden layers with dropout
and a 43 output.
Input:  32x32 image
Return: logits'''

def conv2d(data, weight):
    return tf.nn.depthwise_conv2d_native(data,weight,strides=[1, 1, 1, 1],padding='VALID')    

def max_pool(data):
    return tf.nn.max_pool(data, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

def model(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    dropout = .6
    
    #rgb_to_grayscale
    x = tf.image.rgb_to_grayscale(x, name='grayscale')
    
    x = tf.nn.l2_normalize(x, 1, epsilon=1e-12, name=None)
    #Layer 1: Convolutional. Input = 32x32x1. Output = 26x26x6.
    #out_height = ceil(float(32 - 7 + 1) / float(1)) = 26
    conv1_W = tf.Variable(tf.truncated_normal(shape=(7, 7, 1, 16), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(16))
    conv1   = tf.nn.bias_add(conv2d(x, conv1_W), conv1_b)
    #Pooling. Input = 26x26x5. Output = 13x13x5 out_height = ceil(float(28 - 2 + 1) / float(2)) =  ceil(13.5) = 13
    conv1 = max_pool(conv1)
    conv1  = tf.nn.elu(conv1)   # Activation.
    
    x = tf.nn.l2_normalize(x, 1, epsilon=1e-12, name=None)
    # Layer 2: Convolutional. 1Input = 13, 13, 8 Output = 11, 11, 16.
    #out_height = ceil(float(13 - 3 + 1) / float(1)) = 11
    conv2_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 16, 2), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(32))
    conv2  = conv2d(conv1, conv2_W)+conv2_b
    conv2  = tf.nn.elu(conv2)   # Activation.
    
    x = tf.nn.l2_normalize(x, 1, epsilon=1e-12, name=None)
    conv4_W = tf.Variable(tf.truncated_normal(shape=(1, 1, 32, 1), mean = mu, stddev = sigma))
    conv4_b = tf.Variable(tf.zeros(32))
    conv4  = conv2d(conv2, conv4_W)+conv4_b
    conv4  = tf.nn.elu(conv4)   # Activation.
    
    #Layer 2: Convolutional. 1Input = 11, 11, 30 Output = 8, 8, 35.
    #out_height = ceil(float(8 - 4 + 1) / float(1)) = 8
    conv3_W = tf.Variable(tf.truncated_normal(shape=(4, 4, 32, 2), mean = mu, stddev = sigma))
    conv3_b = tf.Variable(tf.zeros(64))
    conv3  = conv2d(conv4, conv3_W)+conv3_b
    conv3  = max_pool(conv3)    #Pooling. Input = 8, 8, 64. Output = 4, 4, 64.
    conv3  = tf.nn.elu(conv3)   # Activation.

    #Flatten. Input = 4, 4, 64. Output = 1024.
    fc0    = flatten(conv3) 
    
    #Layer 3: Fully Connected. Input = 1024. Output = 688.
    fc1_W  = tf.Variable(tf.truncated_normal(shape=(1024, 688), mean = mu, stddev = sigma))
    fc1_b  = tf.Variable(tf.zeros(688))
    fc1    = tf.matmul(fc0, fc1_W) + fc1_b
    fc1    = tf.nn.dropout(fc1, dropout)
    fc1    = tf.nn.elu(fc1)  # Activation.

    #Layer 4: Fully Connected. Input = 688. Output = 172.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(688, 172), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(172))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    fc2    = tf.nn.dropout(fc2, dropout)
    fc2    = tf.nn.elu(fc2)  # Activation.

    #Layer 5: Fully Connected. Input = 172. Output = 43.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(172, 43), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(43))
    
    weights = fc3_W
    bias    = fc3_b
    
    logits = tf.matmul(fc2, weights) + bias
    
    return logits
