# data_Load.py
# loads stock values from Google
# tail(last item) = most resent date
# head(first item) = oldest date
################################################################################
import matplotlib.dates as mdates
import pandas as pd
import datetime as dt
import random
import pandas_datareader.data as web
from datetime import datetime, timedelta
from os import path
import logging
logger = logging.getLogger()

# import local packages
from helper_Functions import *
from visualize import *

# using a seed to control training data
random.seed(1)
dataLoc = "../data/"

plt.switch_backend('TkAgg')  

#time after algorithm can get new data from Google
timeTreshold = 12

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
        df = pd.read_csv(filePath, encoding = "ISO-8859-1")
    
    return (df['Ticker symbol'].tolist())

##################################################################################
# Downloads data and save it as CSV
# To get fundamentals you would have to pay
# https://www.reddit.com/r/algotrading/comments/4byj5k/is_there_a_python_script_to_get_historical/
# if data is older than a day reload data other wise use the same one from the CSV
# Retunr: the file path where the CSV file is stored
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
        now = dt.datetime.now()
        if (now.hour > 12):
        
            # Define which on-line source one should use
            data_source = 'yahoo'

            # We would like all available data from dataDates[0] until dataDates[1]
            start_date = dataDates[0]
            end_date = todayDate

            # User pandas_reader.data.DataReader to load the desired data. As simple as that.
            try:
                #import pandas_datareader as web
                #df = web.get_data_yahoo('GOOG', start_date, end_date)﻿
                panel_data = web.DataReader(stockName, data_source, start_date, end_date)
                panel_data.to_csv(filePath)
            except:
                print(stockName, "had probles been downloaded")
                if not (path.exists(filePath)):
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

        panel_data.to_csv(filePath)

    else:
        print("file is updated")
               
    return (filePath)

###################################################################################
## reads CSV file into dataFrame
## writes every column in daily % change 
###################################################################################
def readCSV(filePath, days = 0):
    #load csv file
    df = pd.read_csv(filePath, parse_dates = True, index_col=0)

    #append other columns
    df['200ma'] = df['Close'].rolling(window=200).mean()
    df['100ma'] = df['Close'].rolling(window=100).mean()
    df['50ma'] = df['Close'].rolling(window=50).mean()
    df['10ma'] = df['Close'].rolling(window=10).mean()

    if (logger.getEffectiveLevel() == logging.DEBUG):
        #displaying 100ma vs Adj Close
        axBarGraph(index = df.index, ax1Data = df['Adj Close'], \
            ax2Data  = df['100ma'], barData = df['Volume'])

    if (logger.getEffectiveLevel() == logging.DEBUG):
        candlestickGraph(df)

    #append other columns
    if(days):
        df = round(((df - df.shift(days))/df.shift(days) * 100),2)

    #When we reset the index, the old index is added as a column, and a new sequential index is used
    #inplace=True, modefy the current table do not return a new
    df.reset_index(inplace=True)
    return(df)

###################################################################################
## combine CSV files into dataFrame
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
## dataPipeline
## creates a data set, if this is your first time running this code, I would recommend
## delating previews data sets or you may have issues with pickle due to the 
## dispcrepanci in pndas_dataReader verison.
##  
###################################################################################
def dataPipeline(dataDates, stockToPredict):
    # if detaset file already exist do not create new dataset
    filePath = dataLoc + "dataSets"
    if path.exists(filePath):
        print("opening file... ")
        stock_CSVData = openCSV(filePath)
        logging.debug("stock_CSVData", stock_CSVData)

    else:
        #Create/get sp500 name list
        sp500_list = getsp500()

        #get Fundamental data stocks that drive the market
        #petrolium price, usa dollar, gold, 
        #getFundamentalData()

        #Store stock web address it in CSV files
        stockPaths = []
        for item in sp500_list:
            #gets the file path where the CSV file is stored
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


    #load your stock if it hasn't loaded   
    if (not stockToPredict in stock_CSVData):
        stPath = getWebData(stockToPredict, dataDates)
        stock_CSVData.update({stockToPredict: readCSV(stPath)})

    return stock_CSVData


if __name__ == "__main__":
    """ for testing or example purpose 
    """
    import argparse
    # Instantiate the parser
    # logging.basicConfig(filename='log.log', level=logging.INFO)
    parser = argparse.ArgumentParser(description='Optional app description')

    # Debug
    parser.add_argument('-v', "--verbose", action='store_true',
                        help='Type -v to do debugging')

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
    '''
    # logger examples
    # logging.info('So should this')
    # logging.warning('And this, too')
    '''
    #################################################################################

    stockToPredict = 'TSLA'
    dataDates = []
    dataDates.append('2010-01-01')
    dataDates.append(datetime.now().date())
    
    # from csv file with all the stocks from sp500
    # dataPipeline returns a data set to train with 5 random stocks
    df = dataPipeline(dataDates, stockToPredict)
    print(df)
