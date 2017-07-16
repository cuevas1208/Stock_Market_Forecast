# data_prepare.py
# Prepare dataSets for training
# tail(last item) = most resent date
# head(first item) = oldest date
################################################################################
import numpy as np

flag_OneHot = 0

###################################################################################
## round labels
###################################################################################
def roundLabels(y_listNP):
    #Rond down to .5
    y_listNP = np.multiply(y_listNP, 2)
    y_listNP = np.around(y_listNP, decimals=0)
    y_listNP = np.divide(y_listNP, 2)

    #Round to 10 any nuber greater than 10
    y_listNP[y_listNP > 10] = 10

    #Round to -10 any nuber less than -10
    y_listNP[y_listNP < -10] = -10

    return y_listNP

###################################################################################
## round detaset
## rounds to 2 decimals 
###################################################################################
def roundData(y_listNP):
    #Rond down to .5
    y_listNP = np.around(y_listNP, decimals=2)

    #Round to 12 any nuber greater than 12
    y_listNP[y_listNP > 12] = 12

    #Round to -12 any nuber less than -12
    y_listNP[y_listNP < -12] = -12

    return y_listNP

##########################################################################################
# Round data sets
##########################################################################################
def roundDataSet(x_list, y_list):
    #Rond labels
    y_listN = roundLabels(np.array(y_list))
    classesTotal = len(np.unique(y_listN))
    print("unic classes: ", classesTotal)

    #Rond dataSet
    x_listN = roundData(np.array(x_list))
    return x_listN, y_listN

##########################################################################################
# One Hot Encoder
##########################################################################################  
def oneHot(Y):
    from sklearn.preprocessing import LabelEncoder
    from keras.utils import np_utils
    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_y = np_utils.to_categorical(encoded_Y)
    flag_OneHot = 1
    return dummy_y

def splitData(x_listN, y_listN):
    #get the last 30 items for test\validation
    x_last30 = x_listN[-20:]
    y_last30 = y_listN[-20:]
    
    x_listN = x_listN[0:-20]
    y_listN = y_listN[0:-20]

    #Shuffle and split Training, Test, and Validation data
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x_listN, y_listN, test_size=0.25, random_state=42)
    x_test = np.append(x_test,x_last30, axis = 0)
    y_test = np.append(y_test,y_last30, axis = 0)

    x_test, x_valid, y_test, y_valid = train_test_split(x_test, y_test, test_size=0.20, random_state=52)

    assert(len(x_train) == len(y_train))
    assert(len(x_test) == len(y_test))
    assert(len(x_valid) == len(y_valid))

    print ("train dataSet lenght: ", len(x_train))
    print ("test  dataSet lenght: ", len(x_test))
    print ("valid dataSet lenght: ", len(x_valid))

    y_train = roundLabels(np.array(y_train))
    y_test = roundLabels(np.array(y_test))
    y_valid = roundLabels(np.array(y_valid))

    #return traing, test,and validation datatest
    return x_train, x_test, y_train, y_test, x_valid, y_valid


def getclassesTotal():
    if flag_OneHot:
        return classesTotal
    else:
        return 1
