#
#
import csv
import numpy as np
import pandas as pd
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import unicodecsv

plt.switch_backend('TkAgg')  

dates = []
prices = []
stock_File0 = {}
from datetime import datetime as dt

# Takes a date as a string, and returns a Python datetime object. 
# If there is no date given, returns None
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
        '''
        #df = pd.read_csv(filename)
        #dates = df.index.values
        #price = df['price'].values     
        with open(filename, 'r') as csvfile:
            print ("here 1")
            csvFileReader = csv.reader(csvfile)
            print ("here 2")
            next(csvFileReader)	# skipping column names
            print ("here 3")
            for i, row in enumerate(csvFileReader):
                print ("here 3 ", i)
                dates.append(int(row[0].split('-')[0]))
                prices.append(float(row[1]))
                if (i > 29):
                    break
        print (dates)
        print (prices)
        
        '''# Using unicodecsv
        stock_File = read_file(filename)
        # Clean up the data types in the enrollments table
        stock_File0 = stock_File
        print("lines in the file",len(stock_File0))
        for index, stock in enumerate(stock_File0):
            stock['\ufeffDate'] = parse_date(stock['\ufeffDate'])
            stock['Open'] =  float(stock['Open'])
            prices.append(float(stock['Low']))
            stock['High'] = float(stock['High'])
            stock['Close'] = float(stock['Close'])
            dates.append(int(stock['Volume']))

        print (stock_File0[0])
        
        return
        
def predict_price(dates, prices, x):
	dates = np.reshape(dates,(len(dates), 1)) # converting to matrix of n X 1

	svr_lin = SVR(kernel= 'linear', C= 1e3)
	svr_poly = SVR(kernel= 'poly', C= 1e3, degree= 2)
	svr_rbf = SVR(kernel= 'rbf', C= 1e3, gamma= 0.1) # defining the support vector regression models

	#build a model of how the output dates depends on prices
	svr_rbf.fit(dates, prices) # fitting the data points in the models
	svr_lin.fit(dates, prices)
	svr_poly.fit(dates, prices)

	#plt.scatter(dates, prices, color= 'black', label= 'Data') # plotting the initial datapoints 
	plt.plot(dates, svr_rbf.predict(dates), color= 'red', label= 'RBF model') # plotting the line made by the RBF kernel
	plt.plot(dates,svr_lin.predict(dates), color= 'green', label= 'Linear model') # plotting the line made by linear kernel
	plt.plot(dates,svr_poly.predict(dates), color= 'blue', label= 'Polynomial model') # plotting the line made by polynomial kernel
	plt.xlabel('Date')
	plt.ylabel('Price')
	plt.title('Support Vector Regression')
	plt.legend()
	plt.show()

	return svr_rbf.predict(x)[0], svr_lin.predict(x)[0], svr_poly.predict(x)[0]

stock_File0 = get_data('aapl_google.csv') # calling get_data method by passing the csv file to it

#predicted_price = predict_price(dates, prices, 23575094)
#print (predicted_price)











