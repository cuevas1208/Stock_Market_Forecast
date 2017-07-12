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
import pickle


#using a seed to cotrol training data
random.seed(1)
dataLoc = "../data/"

###################################################################################
## HelperFunctions
###################################################################################
###################################################################################
## given two data points it returns persentage
###################################################################################
def persentage(now, whole):
    part = now - whole;
    return 100 * float(part)/float(whole)

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
## openCSV
###################################################################################
def openCSV(filePath):
    dist_pickle = pickle.load(open(filePath, "rb") )
    return dist_pickle["items"]

###################################################################################
## saveCSV
###################################################################################
def saveCSV(filePath, items):
    #Save pre-processed data
    dist_pickle = {}
    dist_pickle["items"] = items
    pickle.dump( dist_pickle, open( filePath, "wb" ))

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
    
    return(main_df)

###################################################################################
## getStockArray
## input: stock
## return np.array
###################################################################################
def getStockArray(stock, sN, i):
    x = np.array([persentage(stock[sN+'_High'][i] ,stock[sN+'_Open'][i]),
                  persentage(stock[sN+'_Low'][i]  ,stock[sN+'_Open'][i]),
                  persentage(stock[sN+'_Close'][i],stock[sN+'_Open'][i]),
                  persentage(stock[sN+'_100ma'][i],stock[sN+'_Open'][i]),
                  persentage(stock[sN+'_200ma'][i],stock[sN+'_Open'][i]),
                  persentage(stock[sN+'_50ma'][i] ,stock[sN+'_Open'][i]),
                  persentage(stock[sN+'_10ma'][i] ,stock[sN+'_Open'][i]),
                  stock['Date'][i]])
    return x

###################################################################################
## getStockArray
## input: stock
## return np.array
###################################################################################
def getStockRArray(stock, sN, i):
    x = np.array([persentage(stock[sN+'_High'][i] ,stock[sN+'_Open'][i]),
                  persentage(stock[sN+'_Low'][i]  ,stock[sN+'_Open'][i]),
                  persentage(stock[sN+'_Close'][i],stock[sN+'_Open'][i]),
                  persentage(stock[sN+'_100ma'][i],stock[sN+'_Open'][i])])
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
    
    for count in range (len(margeCSV)-33):
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

        #if count % 10 == 0:
        #print(count, "out of ", len(margeCSV)-33)

    return dataList, labelList

###################################################################################
## dataPipeline
## data set pipeline
###################################################################################
def dataPipeline(dataDates, stockToPredict):
    #Create an sp500 list
    sp500_list = getsp500()

    #get Fundamental data stocks that drive the market
    #petrolium price, usa dollar, gold, 
    #getFundamentalData()

    #Randomly pick 4 sp500 list
    stocksName = []
    stocksName.append(stockToPredict)
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

    #Join data
    margeCSV = combineCSV(stock_CSVData, stocksName)

    #Create Data sets
    x_list, y_list = createDataSet(margeCSV, stocksName)

    saveCSV(dataLoc + "dataSets", [x_list, y_list])

    return x_list, y_list
    

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
    # if detasetfile already exist do not create new dataset
    filePath = dataLoc + "dataSets"
    if (path.exists(filePath)):
        print ("opening file... ")
        dataSets = openCSV(filePath)
        x_list = dataSets[0]
        y_list = dataSets[1]
        
    else:
        x_list, y_list = dataPipeline(dataDates, stockToPredict)


    #Rond labels
    y_listN = roundLabels(np.array(y_list))
    classesTotal = len(np.unique(y_listN))
    print("unic classes: ", classesTotal)
    

    #Shuffle and split Training, Test, and Validation data
    from sklearn.model_selection import train_test_split

    #get the last 30 items for test\validation
    x_last30 = x_list[-30:]
    y_last30 = y_list[-30:]

    x_train, x_test, y_train, y_test = train_test_split(x_list, y_list, test_size=0.25, random_state=42)
    x_test.extend(x_last30)
    y_test.extend(y_last30)

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
    return x_train, x_test, y_train, y_test, x_valid, y_valid, classesTotal 

###################################################################################
## selfRun
## for testing or example purpose
###################################################################################
if __name__ == "__main__":
    dataDates = []
    dataDates.append('2010-01-01')
    dataDates.append(datetime.now().date())

    get_detaSet(dataDates, 'TSLA')



    
