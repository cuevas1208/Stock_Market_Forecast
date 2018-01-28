# data_Load.py
# loads stock values from Google
# tail(last item) = most resent date
# head(first item) = oldest date
################################################################################
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import datetime as dt
import random, time, html5lib
import pandas_datareader.data as web
from datetime import datetime, timedelta
from os import path
from helper_Functions import *
import time
from tqdm import tqdm
import argparse
import logging
logger = logging.getLogger()
from visualize import *
from data_Load import *

#using a seed to control training data
random.seed(1)
dataLoc = "../data/"

plt.switch_backend('TkAgg')  
###################################################################################
## getStockArray
## input: stock
## return np.array
###################################################################################
def getStockArray(stock, sN, i):
    x = np.array([stock[sN+'_High'][i],
                  stock[sN+'_Low'][i]  ,
                  stock[sN+'_Close'][i],
                  stock[sN+'_100ma'][i],
                  stock[sN+'_200ma'][i],
                  stock[sN+'_50ma'][i] ,
                  stock[sN+'_10ma'][i] ,
                  stock['Date'][i]])
    return x

###################################################################################
## getStockArray
## input: stock
## return np.array
###################################################################################
def getStockRArray(stock, sN, i):
    x = np.array([stock[sN+'_High'][i],
                  stock[sN+'_Low'][i]  ,
                  stock[sN+'_Close'][i],
                  stock[sN+'_100ma'][i]])
    return x

###################################################################################
## convert data into a (32,32,1)
## input: df_ohlc - the main stock to be predicted
## reference stock or external data that can help predict your stock
###################################################################################
def create32by32DataSet(margeCSV, stocksName):

    dataList = []
    labelList = []

    print ("creating 32 by 32 features: ", len(margeCSV)-33)

    for count in tqdm(range (len(margeCSV)-33)):
    
        #creates fist array
        x = np.array (   getStockRArray(margeCSV, stocksName[1], count))
        x = np.append(x, getStockArray (margeCSV, stocksName[0], count))
        x = np.append(x, getStockRArray(margeCSV, stocksName[2], count))
        x = np.append(x, getStockRArray(margeCSV, stocksName[3], count))
        x = np.append(x, getStockArray (margeCSV, stocksName[0], count))
        x = np.append(x, getStockRArray(margeCSV, stocksName[4], count))


        #append to the first array till there is 1024 data  "1024/32 = 32"
        for i in range(1, int(1024/32)):
            x = np.append(x, getStockRArray(margeCSV, stocksName[1], i+count))
            x = np.append(x, getStockArray (margeCSV, stocksName[0], i+count))
            x = np.append(x, getStockRArray(margeCSV, stocksName[2], i+count))
            x = np.append(x, getStockRArray(margeCSV, stocksName[3], i+count))
            x = np.append(x, getStockArray (margeCSV, stocksName[0], i+count))
            x = np.append(x, getStockRArray(margeCSV, stocksName[4], i+count))
            
        x = x.reshape((32, 32, 1))

        #get label
        label = persentage(margeCSV[stocksName[0]+'_Close'][int(1024/32+count)],
                                    margeCSV[stocksName[0]+'_Close'][int(1024/32+count)-1])
        labelList.append(label)
        featureList.append(x)

        time.sleep(0.005)
        #bar.update(count)

    return dataList, labelList

###################################################################################
## getDetaSet
## returns the dataSets: data and labels
## y_list the percentage of the stock changed 
## x_list: are 5 stock that we would like to base our prediction on
## the 5 stocks we would be basing our prediction are random 
###################################################################################
def get_detaSet(dataDates, stockToPredict):
	# get data
	stock_CSVData = dataPipeline(dataDates, stockToPredict)

	#Join data 5 datasets plus the one you selected
	keys = random.sample(list(stock_CSVData), 5)
	keys[0] = stockToPredict
	df_margeCSV, stocksName = combineCSV(stock_CSVData, keys)

	if (logger.getEffectiveLevel() == logging.DEBUG):
		visualize_correlation(df_margeCSV)


	#Create Data sets (32,32,1) mainly for convolution nets
	x_list, y_list = create32by32DataSet(df_margeCSV, stocksName)

	return x_list, y_list

###################################################################################
## selfRun
## for testing or example purpose 
###################################################################################
if __name__ == "__main__":
    import sys
    # Instantiate the parser
    # logging.basicConfig(filename='log.log', level=logging.INFO)
    parser = argparse.ArgumentParser(description='Optional app description')

    #Debug
    parser.add_argument('-v', "--verbose", action='store_true',
                        help='Type -v to do debugging')

    args = parser.parse_args()

    if(args.verbose):
        logger.setLevel(logging.DEBUG)
    '''
    examples
    #logging.info('So should this')
    #logging.warning('And this, too')
    '''
    #################################################################################
    dataDates = []
    dataDates.append('2010-01-01')
    dataDates.append(datetime.now().date())

    get_detaSet(dataDates, 'TSLA')
