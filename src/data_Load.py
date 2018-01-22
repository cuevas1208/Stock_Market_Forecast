# data_Load.py
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

#using a seed to cotrol training data
random.seed(1)
dataLoc = "../data/"

plt.switch_backend('TkAgg')  

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
        try:
            panel_data = web.DataReader(stockName, data_source, start_date, end_date)
            panel_data.to_csv(filePath)
        except:
            print(stockName, "was not found")
            return (0)
               
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
    
    #append other columns
    df = (df.shift(-1) - df)/ df * 100

    #Drop the rows where all of the elements are nan
    main_df = df.dropna(axis=0, how='any')
    
    df.reset_index(inplace=True)
    return(df)

###################################################################################
## reads CSV file into dataFrame
###################################################################################
def combineCSV(CSVfiles, keys):

    main_df = pd.DataFrame()
   
    for key in (keys):
        
        df = CSVfiles[key]  
        df.set_index('Date', inplace=True)
        
        df.drop(['Volume'], 1, inplace=True)
        df = df.add_prefix(key + "_")

        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how='outer')
            
    #Drop the rows where all of the elements are nan
    #main_df = main_df.dropna(axis=0, how='any')

    # the index name will be used as column and numbers will be used.
    main_df.reset_index(inplace=True)

    #convert date format to Month and day 
    main_df['Date'] = main_df['Date'].map(lambda x: (100 * x.month + x.day)/100)

    return(main_df, keys)

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
def createDataSet(margeCSV, stocksName):

    dataList = []
    labelList = []

    print ("detaset length: ", len(margeCSV)-33)
    #with progressbar.ProgressBar(max_value= len(margeCSV)-33) as bar:
    for count in tqdm(range (len(margeCSV)-33)):
    #for count in range (60):   #debugging
    
        #creates fist array
        x = np.array (   getStockRArray(margeCSV, stocksName[1], count))
        x = np.append(x, getStockArray (margeCSV, stocksName[0], count))
        x = np.append(x, getStockRArray(margeCSV, stocksName[2], count))
        x = np.append(x, getStockRArray(margeCSV, stocksName[3], count))
        x = np.append(x, getStockArray (margeCSV, stocksName[0], count))
        x = np.append(x, getStockRArray(margeCSV, stocksName[4], count))


        #append to the first array till there is 1024 data
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
        dataList.append(x)

        time.sleep(0.005)
        #bar.update(count)

    return dataList, labelList


###################################################################################
## dataPipeline
## data set pipeline
###################################################################################
def dataPipeline(dataDates, stockToPredict):
    #if detasetfile already exist do not create new dataset
    filePath = dataLoc + "dataSets"
    #if (path.exists(filePath)):
    if (0):
        print ("opening file... ")
        stock_CSVData = openCSV(filePath)
    
    else:
        #Create an sp500 list
        sp500_list = getsp500()

        #get Fundamental data stocks that drive the market
        #petrolium price, usa dollar, gold, 
        #getFundamentalData()

        #Load stock data from web
        stockPaths = []
        for item in sp500_list:
            stockPath = getWebData(item,dataDates)
            if (stockPath):
                stockPaths.append(stockPath)
            else:
                #removes the first matching value
                sp500_list.remove(item)

        #################################################
        ##Loop to create dataframes
        #################################################
        #Read CSV files and append each other to one data 
        stock_CSVData = {}

        for i, item in enumerate(stockPaths):
            if not stock_CSVData:
                stock_CSVData = ({sp500_list[i]: readCSV(item)})
            else:
                stock_CSVData.update({sp500_list[i]: readCSV(item)})

        saveCSV(dataLoc + "dataSets", stock_CSVData)


    #load your stock       
    if (not stockToPredict in stock_CSVData):
        stPath = getWebData(stockToPredict,dataDates)
        stock_CSVData.update({stockToPredict: readCSV(stPath)})

    return stock_CSVData


###################################################################################
## getDetaSet
## returns the 3 dataSets: traing, test, and validation
## if dataset is updated it it will just load it from a file
## otherwise it will and data new data to the old dataset and train and store data
## input: start: dates to train from start
##        finish: dates to finish train
###################################################################################
def get_detaSet(dataDates, stockToPredict):

    # get data
    stock_CSVData = dataPipeline(dataDates, stockToPredict)

    #Join data
    keys = random.sample(list(stock_CSVData), 5)
    keys[0] = stockToPredict
    df_margeCSV, stocksName = combineCSV(stock_CSVData, keys)

    #Create Data sets
    x_list, y_list = createDataSet(df_margeCSV, stocksName)

    return x_list, y_list


###################################################################################
## selfRun
## for testing or example purpose
###################################################################################
if __name__ == "__main__":
    dataDates = []
    dataDates.append('2010-01-01')
    dataDates.append(datetime.now().date())

    get_detaSet(dataDates, 'TSLA')



    
