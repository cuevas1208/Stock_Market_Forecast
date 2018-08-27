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

from data_functions.data_Load import dataPipeline, combineCSV, getsp500

logger = logging.getLogger()

PREDICTION_DAYS = 4
DATA_SET_LEN = 20
MODELS_PATH = '../models/'

STOCK_TO_PREDICT = 'TSLA'
LABEL_TO_PREDICT = '_High'

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


def process_data_for_labels(ticker_feature, df):
    """ process_data_for_labels
    # input days to analyze data
    # it would output the percentage increase/decrease of the stock in PREDICTION_DAYS
    # by shifting the columns to PREDICTION_DAYS

    :param ticker_feature: string name of the ticker
    :param df: dataset fragment
    :return: df with adjusted percentage change
    """
    print('loading:', ticker_feature, df.shape)
    # delete all NAN array with rows deleted
    df.dropna(subset=[ticker_feature], inplace=True)
    logging.debug("df shape after eliminating NAN from labels", df.shape)

    # get future labels for classification
    df['{}_labels'.format(ticker_feature)] = \
        (df[ticker_feature].shift(-PREDICTION_DAYS) - df[ticker_feature]) / df[ticker_feature]

    # return a list of labels with no NAN
    df.dropna(subset=['{}_labels'.format(ticker_feature)], inplace=True)
    logging.debug("df shape after eliminating NAN from labels", df.shape)
    logging.debug("labels with days difference", df['{}_labels'.format(ticker_feature)])

    # transform labels in to 1, 0 and -1 ==> buy_hold_sell
    df['{}_labels'.format(ticker_feature)] = list(map(set_buy_sell_hold, df['{}_labels'.format(ticker_feature)]))
    logging.debug("labels after the one_hot", df['{}_labels'.format(ticker_feature)])

    # visualize spread
    values = df['{}_labels'.format(ticker_feature)].values.tolist()
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


def extract_features_method_1(df):
    # tickers are all the column_labels that would be for training excluding training_labels
    tickers = df.columns.values.tolist()

    # pct_change() normalize the value to be percentage change
    df_values = df[[ticker for ticker in tickers]].pct_change()

    df_values.replace([np.inf, -np.inf], 0, inplace=True)
    df_values.fillna(0, inplace=True)
    print("x features shape:  ", df_values.shape)

    x = df_values.values

    return x, df


def extract_features_method_2(df, ticker):

    # calculates the medan of low
    df['x_features'] = df[ticker + '_Low'] + df[ticker + '_High'] / 2

    # normalize the value to be percent change
    df['x_features'] = df['x_features'].pct_change()

    # handles invalid numbers
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)

    # extract feature data, data has to be before the label date (shift by one at least)
    x = df['x_features'].shift(1).values
    for i in range(2, DATA_SET_LEN):
        x = np.dstack((x, df['x_features'].shift(i).values))
    x = x[0].tolist()
    df['training_features'] = x

    return df


def get_training_dataset(ticker, df):

    # x, _ = extract_features_method_1(df)
    df = extract_features_method_2(df, ticker)

    # get validation sample before shuffle
    y = df[ticker + LABEL_TO_PREDICT + '_labels'].values
    x = df['training_features'].tolist()
    x = np.asarray(x)
    dates = df['Date'].tolist()
    valid_sample = {'x': x[-120:-90], 'y': y[-120:-90], 'dates': dates[-120:-90]}

    # shuffles dataset
    df = df.sample(frac=1).reset_index(drop=True)

    # load dataset
    y = df[ticker + LABEL_TO_PREDICT + '_labels'].values
    x = df['training_features'].tolist()
    x = np.asarray(x)
    dates = df['Date'].tolist()

    # clean data
    mask = ~np.isnan(x).any(axis=1)
    x = x[mask]; y = y[mask]
    mask = np.isfinite(x).any(axis=1)
    x = x[mask]; y = y[mask]

    print('input training labels shape:', y.shape, ' and features shape:', x.shape)
    return x, y, dates, valid_sample


def train(x, y, valid_sample, ticker='stock'):
    logging.debug("X sample: \ {} ".format(len(x.shape)))
    logging.debug("y sample: \ {} ".format(len(y.shape)))

    # random shuffle and split
    test_size = int(len(y)*0.2)
    x_train, x_test, y_train, y_test = x[test_size:], x[:test_size], y[test_size:], y[:test_size]

    # combine the predictions of several base estimators
    clf = VotingClassifier([('lsvc', svm.LinearSVC()),
                            ('knn', neighbors.KNeighborsClassifier()),
                            ('rfor', RandomForestClassifier())])

    clf.fit(x_train, y_train)

    # test data prediction
    np.set_printoptions(precision=2)
    confidence = clf.score(x_test, y_test)
    print('accuracy:', confidence)

    from visualization import matplot_graphs
    y_pred = clf.predict(valid_sample['x'])
    matplot_graphs.plot_histogram(y_pred, valid_sample['y'], valid_sample['dates'], ticker, str(confidence))

    return confidence, clf


def forecast(ticker, df):
    # load dataset
    x, y, dates, validation_sample = get_training_dataset(ticker, df)

    # load model
    if not os.path.exists(ticker):

        # train model
        confidence, clf = train(x, y, validation_sample, ticker)

        # save model
        if not os.path.exists(MODELS_PATH):
            os.makedirs(MODELS_PATH)
        joblib.dump(clf, MODELS_PATH + ticker + '.pkl')

    # last dataset sample
    x = x[-1:]
    clf = joblib.load(MODELS_PATH + ticker + '.pkl')

    # get forecast
    forecast_ = clf.predict(x)
    print('\n', ticker, ': Based in the last', DATA_SET_LEN, 'market days, the forecast is ',
          forecast_ * PERCENTAGE_THRESHOLD, '% in next', PREDICTION_DAYS, 'days\n')


def get_dataframe(ticker):
    # set training dates range
    # Todo: dingily change the date based on ticker data
    training_range_dates = ['2010-01-01', datetime.now().date()]

    # load get STOCK_TO_PREDICT and 5 random stocks from sp500 to be used as indicators
    df = dataPipeline(training_range_dates, ticker)
    df_features, stocksName = combineCSV(df, df.keys())

    if logger.getEffectiveLevel() == logging.DEBUG:
        logging.debug("Loaded most resent sample: \ {} ".format(df_features.tail(2)))
        df_features[ticker + LABEL_TO_PREDICT].plot()
        plt.show()

    # change values to percentage of change within the days intended to predict
    df = process_data_for_labels(ticker + LABEL_TO_PREDICT, df_features)

    # handle amy invalid numbers
    df.fillna(0, inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    return df


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

    # set debug mode
    if args.verbose: logger.setLevel(logging.DEBUG)

    # forecast chosen ticker
    df = get_dataframe(STOCK_TO_PREDICT)
    forecast(STOCK_TO_PREDICT, df)

    # forecast all sp500
    sp500_list = getsp500()
    for sp500_ticker in sp500_list:
        df = get_dataframe(sp500_ticker)
        forecast(sp500_ticker, df)
