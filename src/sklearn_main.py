# skelearn.py
# Prepare dataSets for training
# techniques learn on the bottom site were implemented
# to build this ml module
# referene  https://pythonprogramming.net/machine-learning-stock-prices-python-programming-for-finance/
################################################################################
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from collections import Counter
from sklearn import svm, cross_validation, neighbors
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
import argparse
import logging
from sklearn.externals import joblib
import os

from data_functions.data_Load import dataPipeline, combineCSV

logger = logging.getLogger()

PREDICTION_DAYS = 7
DATA_SET_LEN = 10
MODELS_PATH = '../models/'
STOCK_TO_PREDICT = 'TSLA'
ticker = STOCK_TO_PREDICT + '_High'

# PERCENTAGE_THRESHOLD should change based on the volatility of the stock
PERCENTAGE_THRESHOLD = 0.02  # 0.02 == 2 percentage


def detaCorrelation(df, stocksName):
    """
    graphs data correlation
    """
    # before doing data correlation it would be good to prepare that data
    # by taking the % differences of the stock market
    df = process_data_for_labels(df)

    df_corr = df.corr()
    # print("correlation: ")
    # print(df_corr)

    data = df_corr.values
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
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
    plt.xticks(rotation=90)
    heatmap.set_clim(-1, 1)
    plt.tight_layout()
    plt.show()


def process_data_for_labels(ticker, df):
    """ process_data_for_labels
    # input days to analyze data
    # it would output the percentage increase/decrease of the stock in PREDICTION_DAYS
    # by shifting the columns to PREDICTION_DAYS

    :param ticker: string name of the ticker
    :param df: dataset fragment
    :return: df with adjusted percentage change
    """
    # delete all NAN array with rows deleted
    df.dropna(subset=[ticker], inplace=True)
    logging.debug("df shape after eliminating NAN from labels", df.shape)

    # get future labels for classification
    df['{}_labels'.format(ticker)] = \
        (df[ticker].shift(-PREDICTION_DAYS) - df[ticker]) / df[ticker]

    # return a list of labels with no NAN
    df.dropna(subset=['{}_labels'.format(ticker)], inplace=True)
    logging.debug("df shape after eliminating NAN from labels", df.shape)

    logging.debug("labels with days difference", df['{}_labels'.format(ticker)])

    # transform labels in to 1, 0 and -1 ==> buy_hold_sell
    df['{}_labels'.format(ticker)] = list(map(set_buy_sell_hold, df['{}_labels'.format(ticker)]))
    logging.debug("labels after the one_hot", df['{}_labels'.format(ticker)])

    # visualize spread
    values = df['{}_labels'.format(ticker)].values.tolist()
    str_values = [str(i) for i in values]
    print('Data spread:', Counter(str_values), "total labels: ", len(values))

    return df


def set_buy_sell_hold(*args):
    """ str_vals is a list of our data labels
        -1  = decrees
        0   = stayed same
        -1  = increased
    """
    cols = [c for c in args]
    for col in cols:
        if col > PERCENTAGE_THRESHOLD:
            return 1
        if col < -PERCENTAGE_THRESHOLD:
            return -1
    return 0


def extract_features(ticker, df):
    """ extract_features
    """
    tickers = df.columns.values.tolist()

    # handles invalid numbers
    df.fillna(0, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)

    # tickers == to all the column labels used for features excluding labels
    # pct_change() normalize the value to be percentage change from yesterday
    df_vals = df[[ticker for ticker in tickers]].pct_change()
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)
    print("x features shape:  ", df_vals.shape)

    x = df_vals.values
    y = df['{}_labels'.format(ticker)].values

    return x, y, df


def train(x, y):
    logging.debug("X sample: \ {} ".format(len(x.shape)))
    logging.debug("y sample: \ {} ".format(len(y.shape)))

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.25)

    # combine the predictions of several base estimators
    clf = VotingClassifier([('lsvc', svm.LinearSVC()),
                            ('knn', neighbors.KNeighborsClassifier()),
                            ('rfor', RandomForestClassifier())])

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    np.set_printoptions(precision=2)
    print('0=>correct,   1,2=>error')
    print(np.abs(np.around((y_pred - y_test), 2)))
    confidence = clf.score(X_test, y_test)
    print('accuracy:', confidence)

    from visualization import matplot_graphs
    matplot_graphs.plot_histogram(y_pred[-20:], y_test[-20:])

    return confidence, clf


def forecast(tickle_name):
    # get the last PREDICTION_DAYS


    # load model
    if os.path.exists(tickle_name):
        clf = joblib.load('tickle_name.pkl')
    else:
        print('file does no exits')


def method1(stockName, df_features):
    x, y, df = extract_features(stockName, df_features)
    confidence, clf = train(x, y)

    # save model
    if not os.path.exists(MODELS_PATH):
        os.makedirs(MODELS_PATH)

    joblib.dump(clf, MODELS_PATH + stockName + '.pkl')


def method2(stockName, df):
    # prepare label
    y = df['{}_labels'.format(ticker)].values

    # ToDo: how to consider the relevant data to train on
    # detaCorrelation(df, stocksName)
    # do_ml(df[stockToPredict]['Close'], df_features)

    # prepare X
    df['x_features'] = df[STOCK_TO_PREDICT + '_Low'] + df[STOCK_TO_PREDICT + '_High'] / 2

    # pct_change() normalize the value to be percent change from yesterday
    df['x_features'] = df['x_features'].pct_change()

    # creating features for prediction
    x = df['x_features'].values
    for i in range(1, DATA_SET_LEN):
        x = np.dstack((x, df['x_features'].shift(-i).values))

    print(x[0])
    x = x[0]
    mask = ~np.isnan(x).any(axis=1)

    # clean dataset
    x = x[mask]
    y = y[mask]

    logging.debug(x.shape)
    logging.debug(y.shape)

    confidence, clf = train(x, y)

    # save model
    if not os.path.exists(MODELS_PATH):
        os.makedirs(MODELS_PATH)

    joblib.dump(clf, MODELS_PATH + stockName + '.pkl')


if __name__ == "__main__":
    import warnings, sys
    if not sys.warnoptions:
        warnings.simplefilter("ignore")

    # parser initial arguments
    # logging.basicConfig(filename='log.log', level=logging.INFO)
    parser = argparse.ArgumentParser(description='Optional app description')
    # run -v for debug mode
    parser.add_argument('-v', "--verbose", action='store_true',
                        help='Type -v to do debugging')
    args = parser.parse_args()
    if args.verbose: logger.setLevel(logging.DEBUG)
    '''
    emaples
    #logging.info('So should this')
    #logging.warning('And this, too')
    '''

    # set training dates range
    training_range_dates = ['2010-01-01', datetime.now().date()]

    # load get STOCK_TO_PREDICT and 5 random stocks from sp500 to be used as indicators
    df = dataPipeline(training_range_dates, STOCK_TO_PREDICT)
    df_features, stocksName = combineCSV(df, df.keys())

    if logger.getEffectiveLevel() == logging.DEBUG:
        logging.debug("Loaded most resent sample: \ {} ".format(df_features.tail(2)))
        df_features[STOCK_TO_PREDICT + '_High'].plot()
        plt.show()

    # change values to percentage of change for the days intended to predict
    df = process_data_for_labels(ticker, df_features)

    # method1(ticker, df)
    method2(ticker, df)
    ###############################################################################
