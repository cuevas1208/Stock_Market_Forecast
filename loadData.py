from pandas_datareader import data
import pandas as pd

#Dowloads data and save it as CSV
'''
# Define which online source one should use
data_source = 'google'

# We would like all available data from 01/01/2000 until 12/31/2016.
start_date = '2010-01-01'
end_date = '2016-12-31'

# User pandas_reader.data.DataReader to load the desired data. As simple as that.
panel_data = data.DataReader('TSLA', data_source, start_date, end_date)

panel_data.to_csv('stock.csv')
'''
df = pd.read_csv('stock.csv', parse_dates = True, index_col=0)

df['100ma'] = df['Close'].rolling(window=100).mean()
df.dropna(inplace=True)

print(df.head())
                  
#print (type(panel_data))
#print (close.head())
#print (close.describe())
