## File: helper_Functions.py
## Name: Manuel Cuevas
## Date: 01/14/2017
## Project: CarND - LaneLines
## Desc: This is a document to add all functions that are not related to the model
##       but helpful to document or visualize data
## This project was part of the CarND program.
## Tools learned in class were used to identify lane lines on the road.
## Revision: Rev 0000.004
#####################################################################################
#importing useful packages
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from numpy.random import normal, uniform
import numpy as np

"""printImg
Prints two images to compare input and output of a function
Input: img1,img2 - input images to display
       img1_title, img2_title - headers for image one and two
Returns: None                                                            """
def printImg(img1, img2, img1_title = 'Input Image', img2_title = 'Output Image'):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img1)
    ax1.set_title(img1_title, fontsize=50)
    ax2.imshow(img2)
    ax2.set_title(img2_title, fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

"""dataSetInfo
Outout general information about the deta set, and displays a graph with the data distibution
Input: X_train, y_train, X_test, y_test, X_valid, y_valid - Images and Labels for each data set
Returns: None                                                            """    
def dataSetInfo(X_train, y_train, X_test, y_test, X_valid, y_valid):
    n_train = len(X_train)
    n_test = len(X_test)
    image_shape = str(format(X_train[0].shape))
    classes, counts = np.unique(y_train, return_counts = True)
    n_classes = len(classes)

    print("Image Shape: {}".format(X_train[0].shape))
    print("Number of training samples =", n_train)
    print("Number of validation samples =", len(X_valid), (len(X_valid)*100)/n_train,"% of training data")
    print("Number of testing samples =", n_test)
    print("Image data shape =", image_shape)
    print("Number of classes =", n_classes)
    
    #Visualize data set distribution
    bins = n_classes
    plt.hist(y_test, bins=bins, histtype='stepfilled', color='b', alpha=.5, label='Test')
    plt.hist(y_train, bins=bins, histtype='stepfilled', color='r', alpha=.5, label='Training')
    plt.hist(y_valid, bins=bins, histtype='stepfilled', color='g', alpha=.5, label='Validation')

    plt.title("Labels Histogram")
    plt.xlabel("Lable")
    plt.ylabel("Quantity")
    plt.axis([0, 43,  0, counts[0]+1000])

    plt.legend()
    plt.show()
