# skelearn.py
# Prepare dataSets for training
# techniques learn on the bottom site were implemented
# to build this ml module
# referene  https://pythonprogramming.net/machine-learning-stock-prices-python-programming-for-finance/
################################################################################
import numpy as np
from data_Load import dataPipeline, combineCSV
import datetime as dt
from datetime import datetime, timedelta
from matplotlib import style
import matplotlib.pyplot as plt
import pandas as pd
import bs4 as bs
from collections import Counter
import pickle, requests
from sklearn import svm, cross_validation, neighbors
from sklearn.ensemble import VotingClassifier, RandomForestClassifier

##########################################################
#Data correlation
##########################################################
def detaCorrelation(df, stocksName):
    # before doing data correlation it would be good to prepare that data
    # by taking the % differences of the stock marquet
    df = process_data_for_labels(df)
    
    df_corr = df.corr()
    #print("correlation: ") 
    #print(df_corr)

    data = df_corr.values
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    heatmap = ax.pcolor(data, cmap=plt.cm.RdYlGn)
    fig.colorbar(heatmap)
    ax.set_xticks(np.arange(data.shape[0]) + 0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[1]) + 0.5, minor=False)
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    column_labels = df_corr.columns
    row_labels = df_corr.index

    ax.set_xticklabels(column_labels)
    ax.set_yticklabels(row_labels)
    plt.xticks(rotation = 90)
    heatmap.set_clim(-1,1)
    plt.tight_layout()
    plt.show()

###################################################################################
# process_data_for_labels
# input days to analize data 
###################################################################################
def process_data_for_labels(ticker, df):
    hm_days = 7
    tickers = df.columns.values.tolist()
    df.fillna(0, inplace=True)

    print(len(df.columns))
    
    for i in range(1,hm_days+1):
        df['{}_{}d'.format(ticker,i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]
    print("after")
    print(len(df.columns))
        
    df.fillna(0, inplace=True)
    return tickers, df

###################################################################################
# set the classifier
###################################################################################
def buy_sell_hold(*args):
    cols = [c for c in args]
    requirement = 0.02
    for col in cols:
        if col > requirement:
            return 1
        if col < -requirement:
            return -1
    return 0

###################################################################################
# extract_featuresets
###################################################################################
def extract_featuresets(ticker, df_features):
    tickers, df = process_data_for_labels(ticker, df_features)

    df['{}_target'.format(ticker)] = list(map( buy_sell_hold,
                                               df['{}_1d'.format(ticker)],
                                               df['{}_2d'.format(ticker)],
                                               df['{}_3d'.format(ticker)],
                                               df['{}_4d'.format(ticker)],
                                               df['{}_5d'.format(ticker)],
                                               df['{}_6d'.format(ticker)],
                                               df['{}_7d'.format(ticker)] ))


    vals = df['{}_target'.format(ticker)].values.tolist()
    str_vals = [str(i) for i in vals]
    print('Data spread:',Counter(str_vals), "total labels: ", len(vals))

    #handles invalid numbers
    df.fillna(0, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)

    #tickers == to all the column labels used for features excluding labels
    df_vals = df[[ticker for ticker in tickers]].pct_change()
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)
    print("features ", " shape:  ", df_vals.shape)
    
    X = df_vals.values
    y = df['{}_target'.format(ticker)].values
    
    return X,y,df

###################################################################################
# trainData
###################################################################################
def do_ml(ticker, df_features):
    X, y, df = extract_featuresets(ticker, df_features)

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,
                                                        y,
                                                        test_size=0.25)
    
    #combine the predictions of several base estimators
    clf = VotingClassifier([('lsvc',svm.LinearSVC()),
                            ('knn',neighbors.KNeighborsClassifier()),
                            ('rfor',RandomForestClassifier())])

    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    print('accuracy:',confidence)
    predictions = clf.predict(X_test)
    print('predicted class counts:',Counter(predictions))
    print()

    return confidence

###################################################################################
# main
###################################################################################
if __name__ == "__main__":
    stockToPredict = 'TSLA'
    
    dataDates = []
    dataDates.append('2010-01-01')
    dataDates.append(datetime.now().date())
    
    df = dataPipeline(dataDates, stockToPredict)
    df_features, stocksName = combineCSV(df, df.keys())
    print(df_features["Date"].head())
  
    #detaCorrelation(df, stocksName)
    #do_ml(df[stockToPredict]['Close'], df_features)
    do_ml(stockToPredict+'_Close', df_features)
    ###########################################################
