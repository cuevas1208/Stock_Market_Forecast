import matplotlib.dates as mdates
from pandas_datareader import data
import pandas as pd
import numpy as np

##################################################################################
#Dowloads data and save it as CSV
##################################################################################
def getWebData(): 
    # Define which online source one should use
    data_source = 'google'

    # We would like all available data from 01/01/2000 until 12/31/2016.
    start_date = '2010-01-01'
    end_date = '2016-12-31'

    # User pandas_reader.data.DataReader to load the desired data. As simple as that.
    panel_data = data.DataReader('TSLA', data_source, start_date, end_date)

    print(panel_data.head())
    panel_data.to_csv('stock.csv')

'''
#################################################################################
def parse_date(date):
    if date == '':
        return None
    else:
        try:
            return int(time.mktime(dt.strptime(date, '%d-%b-%y').timetuple()))
        except:
            return date

def read_file(filename):
    with open(filename, 'rb') as f:
        reader = list(unicodecsv.DictReader(f))
        return reader

def get_data(filename):
    stock_File = read_file(filename)
    # Clean up the data types in the enrollments table
    stock_File0 = stock_File
    print("lines in the file",len(stock_File0))
    for index, stock in enumerate(stock_File0):
        stock['Date'] = parse_date(stock['Date'])
        stock['Open'] =  float(stock['Open'])
        #prices.append(float(stock['Low']))
        stock['High'] = float(stock['High'])
        stock['Close'] = float(stock['Close'])
        #dates.append(int(stock['Volume']))

    print (stock_File0[0])

    return

get_data("stock.csv")
##################################################################################
'''

###################################################################################
## reads CSV file into dataFrame
###################################################################################
def readCSV():
    df = pd.read_csv('stock.csv', parse_dates = True, index_col=0)
    df['100ma'] = df['Close'].rolling(window=100).mean()

    #eleminates NA data
    df.dropna(inplace=True)
    print(type(df.head()))
    print(df.head())

    #video p.4
    df_ohlc = df['Close'].resample('10D').ohlc()

    df_ohlc.reset_index(inplace=True)
    df_ohlc['Date'] = df_ohlc['Date'].map(mdates.date2num)
    print(df_ohlc.head())
    return(df_ohlc)

###################################################################################
## convert data into a (32,32,1)
###################################################################################
def createDataSet(df_ohlc):
    #creates fist array
    x = np.array([df_ohlc['high'][1]  - df_ohlc['open'][1],
                  df_ohlc['low'][1]   - df_ohlc['open'][1],
                  df_ohlc['close'][1] - df_ohlc['open'][1],
                  df_ohlc['close'][1] - df_ohlc['open'][1]])
    print(x.shape)


    #append to the first array till there is 1024 data
    for i in range(int(1024/4)-1):
        x = np.append(x, [df_ohlc['high'][1] - df_ohlc['open'][1],
                          df_ohlc['low'][1] - df_ohlc['open'][1],
                          df_ohlc['close'][1] - df_ohlc['open'][1],
                          df_ohlc['close'][1] - df_ohlc['open'][1]])
    print(x.shape)

    x = x.reshape((32, 32, 1))
    print(x.shape)

###################################################################################
## main
###################################################################################
df_ohlc = readCSV()
createDataSet(df_ohlc)
