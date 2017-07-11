#importing useful packages
from matplotlib import pyplot as plt
import random
import numpy as np
import tensorflow as tf
import tqdm
import time
import os.path
import pickle

from helper_Functions import *
from model_calls import *
from model_architecture import *
from loadData import *

#%matplotlib inline


if __name__ == "__main__":
    #load data
    dataDates = []
    dataDates.append('2010-01-01')
    dataDates.append(datetime.now().date())

    x_train, x_test, y_train, y_test, x_valid, y_valid = get_detaSet(dataDates, 'TSLA')

    #training model
    saver = training_model(x_train, y_train, x_valid, y_valid, rate = 0.001)

    import tensorflow as tf
    save_file = './saved_models/model.ckpt'

    saver = tf.train.import_meta_graph(save_file + '.meta')

    with tf.Session() as sess:
        saver.restore(sess, save_file)
        test_accuracy = evaluate(x_test, y_test)
        print("Test Accuracy = {:.3f}".format(test_accuracy))
