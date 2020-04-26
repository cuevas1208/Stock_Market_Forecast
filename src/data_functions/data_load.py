# data_load.py
# loads stock values from Google
# tail(last item) = most resent date
# head(first item) = oldest date
################################################################################
import os

import pandas as pd
import datetime as dt
import random
import pandas_datareader.data as web
from datetime import datetime, timedelta
from os import path
import logging
from tqdm import tqdm

from matplotlib import pyplot as plt

from src.conf import START_TIME, STOCK_TO_PREDICT, DATA_LOC, FULL_DATA_PKL, STOCK_INDEX_LOC
# plt.switch_backend('TkAgg')
# from src.sklearn_main import logger
# logger = logging.getLogger()

# import local packages
from src.helper_functions import visualize as vis
from src.helper_functions import helper_Functions as hf

# using a seed to control training data
random.seed(1)


def get_dataframe(ticker='TSLA'):
    """
    :param ticker: string
    :return: Returns a dataframe with all stocks data from the spy500
    """
    # return pd.read_pickle(FULL_DATA_PKL)
    # if os.path.isfile(False): # forces to load dataset
    if os.path.isfile(FULL_DATA_PKL):

        # update file after 3pm pacific time
        fileDate = datetime.fromtimestamp(path.getmtime(FULL_DATA_PKL))
        todayDate = datetime.now()
        if (todayDate.hour < 13 and fileDate.date() >= todayDate.date() - timedelta(days=1)) or \
                (fileDate.hour >= 13 and fileDate.date() == todayDate.date()):
            df_features = pd.read_pickle(FULL_DATA_PKL)
            if ticker + '_Close' in df_features.columns or ticker == None:
                return df_features

    # set training dates range
    training_range_dates = [START_TIME, datetime.now().date()]

    # loads all the in to dataframe along with their moving average in to a dictionary of stocks
    df = dataPipeline(training_range_dates, ticker)

    # combines all stocks in to one dataframe
    df_features, _ = combineCSV(df, df.keys())
    df_features.to_pickle(FULL_DATA_PKL)

    # if logger.getEffectiveLevel() == logging.DEBUG:
    #     logging.debug("Loaded most resent sample: \ {} ".format(df_features.tail(2)))
    #     df_features[ticker + LABEL_TO_PREDICT].plot()
    #     plt.show()

    return df_features


def getsp500():
    """ list all sp500
    return a list a list from sp500
    checks to see if list has been exists if not it would be created
    """
    if not (path.exists(STOCK_INDEX_LOC)):
        df = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
        df.columns = df.ix[0]
        df.drop(df.index[0], inplace=True)
        df['Ticker symbol'].to_csv(STOCK_INDEX_LOC, index=False)
    else:
        # load list from CSV file
        # https://stackoverflow.com/questions/19699367/unicodedecodeerror-utf-8-codec-cant-decode-byte
        df = pd.read_csv(STOCK_INDEX_LOC, encoding="ISO-8859-1")

    return df['Ticker symbol'].tolist()


def getWebData(stockName, dataDates):
    """"
    Downloads data and save it as CSV
    To get fundamentals you would have to pay
    https://www.reddit.com/r/algotrading/comments/4byj5k/is_there_a_python_script_to_get_historical/
    if data is older than a day reload data other wise use the same one from the CSV
    Retunr: the file path where the CSV file is stored
    """

    if stockName == None:
        return 0
    # refresh files only if they haven't done within the day
    filePath = DATA_LOC + stockName + '.csv'
    todayDate = dataDates[1]
    if path.exists(filePath):
        fileDate = datetime.fromtimestamp(path.getmtime(filePath))
        now = dt.datetime.now()

        # if file is from today after 1pm pacific time
        if (now.date() == fileDate.date()) and (fileDate.hour >= 13):
            return filePath

    # Define which on-line source one should use
    data_source = 'yahoo'

    # We would like all available data from dataDates[0] until dataDates[1]
    start_date = dataDates[0]
    end_date = todayDate

    # User pandas_reader.data.DataReader to load the desired data. As simple as that.
    try:
        # import pandas_datareader as web
        # df = web.get_data_yahoo('GOOG', start_date, end_date)﻿
        panel_data = web.DataReader(stockName, data_source, start_date, end_date)
        panel_data.to_csv(filePath)
    except Exception as e:
        print(stockName, 'error while loading \n', e)
        if not (path.exists(filePath)):
            return 0

    return filePath


def getFundamentalData(stock_name="FRED/GDP"):
    """
    # Dowloads data and save it as CSV
    # to get fundamentals you would have to pay
    # https://www.reddit.com/r/algotrading/comments/4byj5k/is_there_a_python_script_to_get_historical
    # in my I will be using quandl
    # The Quandl Python module is free. If you would like to make more than 50 calls a day,
    # however, you will need to create a free Quandl account and set your API key.
    """

    # refresh files only if they haven't done within the day
    fileName = stock_name.replace("/", "_")
    filePath = DATA_LOC + fileName + '.csv'

    # refresh data ones a day
    todayDate = datetime.now().date()

    # if file exist, get files Data. Else stamp old date
    if path.exists(filePath):
        fileDate = datetime.fromtimestamp(path.getctime(filePath)).date()
    else:
        fileDate = datetime.now().date() - timedelta(days=1)

    if todayDate > fileDate:

        # We would like all available data from 01/01/2000 until 12/31/2016.
        start_date = START_TIME
        end_date = todayDate

        import quandl
        # User pandas_reader.data.DataReader to load the desired data. As simple as that.
        panel_data = quandl.get(stock_name, start_date="2001-12-31", end_date=todayDate)

        panel_data.to_csv(filePath)

    else:
        print("file is updated")

    return filePath


def readCSV(filePath, days=0):
    """ reads CSV file into dataFrame
    writes every column in daily % change
    """

    # load csv file
    df = pd.read_csv(filePath, parse_dates=True, index_col=0)

    # append other columns
    df['200ma'] = df['Close'].rolling(window=200).mean()
    df['100ma'] = df['Close'].rolling(window=100).mean()
    df['50ma'] = df['Close'].rolling(window=50).mean()
    df['20ma'] = df['Close'].rolling(window=20).mean()

    # if (logger.getEffectiveLevel() == logging.DEBUG):
    #     #displaying 100ma vs Adj Close
    #     vis.axBarGraph(index = df.index, ax1Data = df['Adj Close'], \
    #         ax2Data  = df['100ma'], barData = df['Volume'])

    # if (logger.getEffectiveLevel() == logging.DEBUG):
    #     vis.candlestickGraph(df)

    # append other columns
    if days:
        df = round(((df - df.shift(days)) / df.shift(days) * 100), 2)

    # When we reset the index, the old index is added as a column, and a new sequential index is used
    # inplace=True, modefy the current table do not return a new
    df.reset_index(inplace=True)
    return (df)


def combineCSV(CSVfiles, keys, main_df=pd.DataFrame()):
    """ combine CSV files into one dataFrame
        if maid_df is provided, keys would be appended it
    """

    for key in keys:

        df = CSVfiles[key]
        df.set_index('Date', inplace=True)

        df.drop(['Volume'], 1, inplace=True)
        df = df.add_prefix(key + "_")

        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how='outer')

    # Drop the rows where all of the elements are nan
    # main_df = main_df.dropna(axis=0, how='any')

    # the index name will be used as column and numbers will be used.
    main_df.reset_index(inplace=True)

    # convert date format to Month and day
    main_df['Date'] = main_df['Date'].map(lambda x: (100 * x.month + x.day) / 100)

    return main_df, keys


def dataPipeline(data_dates, stock_to_predict):
    """ dataPipeline
    creates a data set, if you have issues loading dataset I would recommend
    deleting previews data sets pandas_dataReader verision may have a conflict
    :return a dictionary of all datasets from spy, each row is a dataframe of a dataset
    """

    # if detaset file already exist do not create new dataset
    filePath = DATA_LOC + "dataSets"
    if False:  # path.exists(filePath):
        print("opening file... ")
        stock_CSVData = hf.openCSV(filePath)
        logging.debug("stock_CSVData", stock_CSVData)
    else:
        # Create/get sp500 name list
        sp500_list = getsp500()

        # Store stock web address it in CSV files
        stockPaths = []
        print('loading data from the web ... ')
        for item in tqdm(sp500_list):

            # gets the file path where the CSV file is stored
            stockPath = getWebData(item, data_dates)
            if stockPath:
                stockPaths.append(stockPath)
            else:
                # removes the first matching value
                sp500_list.remove(item)

        # Read CSV files and append each other to one dataframe
        tmp = pd.DataFrame(sp500_list, columns=['Ticker symbol'])
        tmp['Ticker symbol'].to_csv(STOCK_INDEX_LOC, index=False)
        stock_CSVData = {}

        for i, item in enumerate(stockPaths):
            if not stock_CSVData:
                stock_CSVData = ({sp500_list[i]: readCSV(item)})
            else:
                stock_CSVData.update({sp500_list[i]: readCSV(item)})

        hf.saveCSV(DATA_LOC + "dataSets", stock_CSVData)

    # load your stock if it hasn't loaded
    if not stock_to_predict in stock_CSVData:
        stPath = getWebData(stock_to_predict, data_dates)
        if os.path.isfile(stPath):
            stock_CSVData.update({stock_to_predict: readCSV(stPath)})

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

    # if args.verbose:
    #     logger.setLevel(logging.DEBUG)
    '''
    # logger examples
    # logging.info('So should this')
    # logging.warning('And this, too')
    '''
    #################################################################################

    stockToPredict = STOCK_TO_PREDICT
    dataDates = []
    dataDates.append(START_TIME)
    dataDates.append(datetime.now().date())

    # from csv file with all the stocks from sp500
    # dataPipeline returns a data set to train with 5 random stocks
    df = dataPipeline(dataDates, stockToPredict)
    print(df)
