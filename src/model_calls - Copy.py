# loadData.py
# tail(last item) = most resent date
# head(first item) = oldest date
################################################################################
import html5lib
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import datetime as dt
import random, time
import pandas_datareader.data as web
from datetime import datetime, timedelta
from os import path


#using a seed to cotrol training data
random.seed(1)

###################################################################################
## HelperFunctions
###################################################################################
###################################################################################
## given two data points it returns persentage
###################################################################################
def persentage(now, whole):
    part = now - whole;
    return 100 * float(part)/float(whole)

##################################################################################
#  list all sp500
#  return a list a list from sp500
#  checks to see if list has been exists if not it would be created 
##################################################################################
def getsp500():
    filePath = "../data/" + "sp500_list" + '.csv'
    if not (path.exists(filePath)):
        df = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
        df.columns = df.ix[0]
        df.drop(df.index[0], inplace=True)
        df.to_csv(filePath)
    else:
        #load list from CSV file
        #https://stackoverflow.com/questions/19699367/unicodedecodeerror-utf-8-codec-cant-decode-byte
        df = pd.read_csv(filePath,encoding = "ISO-8859-1")
    
    return (df['Ticker symbol'].tolist())

##################################################################################
# Dowloads data and save it as CSV
# to get fundamentals you would have to pay
# https://www.reddit.com/r/algotrading/comments/4byj5k/is_there_a_python_script_to_get_historical/
##################################################################################
def getWebData(stockName, dataDates):

    #refresh files only if they haven't done within the day
    filePath = "../data/" + stockName + '.csv'
    
    #refresh data ones a day
    todayDate = dataDates[1]

    #if file exist, get files modification data. Else stamp old date
    if (path.exists(filePath)):
        fileDate = datetime.fromtimestamp(path.getmtime(filePath)).date()
    else:
        fileDate = datetime.now().date() - timedelta(days=1)
        
    if todayDate > fileDate:
        
        # Define which online source one should use
        data_source = 'google'

        # We would like all available data from dataDates[0] until dataDates[1]
        start_date = dataDates[0]
        end_date = todayDate

        # User pandas_reader.data.DataReader to load the desired data. As simple as that.
        panel_data = web.DataReader(stockName, data_source, start_date, end_date)

        #f = web.DataReader("F", 'yahoo-dividends', start_date, end_date)

        #print(f.head())
        print(panel_data.head())

        panel_data.to_csv(filePath)

    else:
        print("file is updated")
               
    return (filePath)

##################################################################################
# Dowloads data and save it as CSV
# to get fundamentals you would have to pay
# https://www.reddit.com/r/algotrading/comments/4byj5k/is_there_a_python_script_to_get_historical
# in my I will be using quandl
# The Quandl Python module is free. If you would like to make more than 50 calls a day,
# however, you will need to create a free Quandl account and set your API key.
##################################################################################
def getFundamentalData(stockName = "FRED/GDP"):

    #refresh files only if they haven't done within the day
    fileName = stockName.replace("/", "_")
    filePath = "../data/" + fileName + '.csv'

    #refresh data ones a day
    todayDate = datetime.now().date() 

    #if file exist, get files Data. Else stamp old date
    if (path.exists(filePath)):
        fileDate = datetime.fromtimestamp(path.getctime(filePath)).date()
    else:
        fileDate = datetime.now().date() - timedelta(days=1)
        
    if todayDate > fileDate:

        # We would like all available data from 01/01/2000 until 12/31/2016.
        start_date = '2010-01-01'
        end_date = todayDate

        import quandl
        # User pandas_reader.data.DataReader to load the desired data. As simple as that.
        panel_data = quandl.get(stockName, start_date="2001-12-31", end_date = todayDate)

        print(panel_data.head())

        panel_data.to_csv(filePath)

    else:
        print("file is updated")
               
    return (filePath)


###################################################################################
## reads CSV file into dataFrame
###################################################################################
def readCSV(filePath):
    #load csv file 
    df = pd.read_csv(filePath, parse_dates = True, index_col=0)

    #append other columns
    df['200ma'] = df['Close'].rolling(window=200).mean()
    df['100ma'] = df['Close'].rolling(window=100).mean()
    df['50ma'] = df['Close'].rolling(window=50).mean()
    df['10ma'] = df['Close'].rolling(window=10).mean()

    #video p.4
    #df_ohlc = df['Close'].resample('10D').ohlc()
    
    df.reset_index(inplace=True)
    return(df)

###################################################################################
## reads CSV file into dataFrame
###################################################################################
def combineCSV(CSVfiles, stocksName):

    main_df = pd.DataFrame()
   
    for count, df in enumerate(CSVfiles):
        
        df.set_index('Date', inplace=True)
        
        df.drop(['Volume'], 1, inplace=True)
        df = df.add_prefix(stocksName[count]+"_")

        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how='outer')

        if count % 10 == 0:
            print(count)

    #Drop the rows where all of the elements are nan
    main_df = main_df.dropna(axis=0, how='any')

    # the index name will be used as column and numbers will be used.
    main_df.reset_index(inplace=True)

    #convert date format to Month and day 
    main_df['Date'] = main_df['Date'].map(lambda x: 100 * x.month + x.day)
    print (main_df.head())
    
    return(main_df)

###################################################################################
## getStockArray
## input: stock
## return np.array
###################################################################################
def getStockArray(stock, i, sN):
    x = np.array([persentage(stock[sN[i]+'_High'][i] ,stock[sN[i]+'_Open'][i]),
                  persentage(stock[sN[i]+'_Low'][i]  ,stock[sN[i]+'_Open'][i]),
                  persentage(stock[sN[i]+'_Close'][i],stock[sN[i]+'_Open'][i]),
                  persentage(stock[sN[i]+'_100ma'][i],stock[sN[i]+'_Open'][i]),
                  persentage(stock[sN[i]+'_200ma'][i],stock[sN[i]+'_Open'][i]),
                  persentage(stock[sN[i]+'_50ma'][i] ,stock[sN[i]+'_Open'][i]),
                  persentage(stock[sN[i]+'_10ma'][i] ,stock[sN[i]+'_Open'][i]),
                  stock[sN[i]+'_Date'][i]])
    return x

###################################################################################
## getStockArray
## input: stock
## return np.array
###################################################################################
def getStockRArray(stock, i, sN):
    x = np.array([persentage(stock[sN[i]+'_High'][i] ,stock[sN[i]+'_Open'][i]),
                  persentage(stock[sN[i]+'_Low'][i]  ,stock[sN[i]+'_Open'][i]),
                  persentage(stock[sN[i]+'_Close'][i],stock[sN[i]+'_Open'][i]),
                  persentage(stock[sN[i]+'_100ma'][i],stock[sN[i]+'_Open'][i])])
    return x

###################################################################################
## convert data into a (32,32,1)
## input: df_ohlc - the main stock to be predicted
## reference stock or external data that can help predict your stock
###################################################################################
def createDataSet(stockData, stocksName):
    #creates fist array
    x = np.array (   getStockRArray(stockData[1], 0, stocksName))
    x = np.append(x, getStockArray (stockData[0], 0, stocksName))
    x = np.append(x, getStockRArray(stockData[2], 0, stocksName))
    x = np.append(x, getStockRArray(stockData[3], 0, stocksName))
    x = np.append(x, getStockArray (stockData[0], 0, stocksName))
    x = np.append(x, getStockRArray(stockData[4], 0, stocksName))


    #append to the first array till there is 1024 data
    for i in range(1, int(1024/32)):
        x = np.append(x, getStockRArray(stockData[1], i, stocksName))
        x = np.append(x, getStockArray(stockData[0] , i, stocksName))
        x = np.append(x, getStockRArray(stockData[2], i, stocksName))
        x = np.append(x, getStockRArray(stockData[3], i, stocksName))
        x = np.append(x, getStockArray(stockData[0] , i, stocksName))
        x = np.append(x, getStockRArray(stockData[4], i, stocksName))
        
    print("data size",x.shape)
    x = x.reshape((32, 32, 1))
    print("data shape", x.shape)

    #get label
    label = stockData[0]['Close'][int(1024/32)]

    return x, label

###################################################################################
## dataPipeline
## data set pipeline
###################################################################################
def dataPipeline(dataDates):
    #Create an sp500 list
    sp500_list = getsp500()

    #get Fundamental data stocks that drive the market
    #petrolium price, usa dollar, gold, 
    #getFundamentalData()

    #Randomly pick 4 sp500 list
    stocksName = []
    stocksName.append('TSLA')
    stocksName.extend(random.sample(sp500_list, 4))

    #Load stock data from web
    stockPaths = []
    for item in stocksName:
        stockPaths.append(getWebData(item,dataDates))
    print(stockPaths)


    #################################################
    ##Loop to create dataframes
    #################################################
    #Read CSV files and append each other to one data 
    stock_CSVData = []
    for item in stockPaths: 
        stock_CSVData.append(readCSV(item))
        
    combineCSV(stock_CSVData, stocksName)

    dataSetBatch = createDataSet(stock_CSVData, stocksName)
'''    
    print (stock_CSVData[0]['Date'][0])
    print (stock_CSVData[1]['Date'][0])
    
    dataSetBatch = []

    for i in range (100):
        #32x32x1
        dataSetBatch.append(createDataSet(stockCSV_Data))

    return 0
'''
###################################################################################
## getDetaSet
## returns the 3 dataSets: traing, test, and validation
## if dataset is updated it it will just load it from a file
## otherwise it will and data new data to the old dataset and train and store data
## input: start: dates to train from start
##        finish: dates to finish train
###################################################################################

#find out if data has been store and it is updated

#get detaSet
dataDates = []
dataDates.append('2010-01-01')
dataDates.append(datetime.now().date())

dataPipeline(dataDates)


#store 3dataSets traing, test, validation in one file

