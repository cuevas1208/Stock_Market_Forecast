# skelearn.py
# Prepare dataSets for training
# techniques learn on the bottom site were implemented
# to build this ml module
# referene  https://pythonprogramming.net/machine-learning-stock-prices-python-programming-for-finance/
################################################################################
import argparse
import logging
import os
from collections import Counter

import pandas as pd
import numpy as np
from sklearn import svm, neighbors
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.externals import joblib

from src.correlation_test import get_correlation
from src.conf import PREDICTION_DAYS, BATCH_LEN, MODELS_PATH, STOCK_TO_PREDICT, LABEL_TO_PREDICT, \
    PERCENTAGE_THRESHOLD
from src.data_functions.data_load import get_dataframe

logger = logging.getLogger()


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

    # get future labels for classification, shift data
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

    # handle any invalid numbers
    df.fillna(0, inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

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

    # get correlation stock
    correlation_stock = get_correlation(df, ticker + LABEL_TO_PREDICT, top_samples=1, set_len=12).index

    def shift_data(corr, pct=False):
        if pct:
            df[corr] = df[corr].pct_change()
        x = df[corr].shift(0).values
        for i in range(1, BATCH_LEN):
            x = np.dstack((x, df[corr].shift(i).values))
        return x

    # calculates the medan of low
    df['x_features'] = df[ticker + LABEL_TO_PREDICT]  # df[ticker + '_Low'] + df[ticker + '_High'] / 2

    # normalize the value to be percent change
    df['x_features'] = df['x_features'].pct_change()
    print('average percentage change {} in the last 20 days {}\n\n'.format(np.abs(df['x_features']).mean(),
                                                                           np.abs(df['x_features'][-20:]).mean()))

    # extract past feature data including today's data if known
    # Todo: currently row goes from new to old
    x = shift_data('x_features')

    for corr in correlation_stock:
        new_x = (shift_data(corr, pct=True))
        x = np.concatenate([x, new_x], -1)
    x = x[0].tolist()
    df['training_features'] = x

    # handles invalid numbers
    # df.replace([np.inf, -np.inf], 0, inplace=True)
    # df.fillna(0, inplace=True)

    return df


def get_training_dataset(ticker, df):
    # change values to percentage of change within the days intended to predict
    df = process_data_for_labels(ticker + LABEL_TO_PREDICT, df)

    # x, _ = extract_features_method_1(df)
    df = extract_features_method_2(df, ticker)

    # take the last 7 days as validation samples before shuffle
    # this is mainly for visualization
    sample_size = 15
    valid = df[-sample_size:]
    y = valid[ticker + LABEL_TO_PREDICT + '_labels'].values
    x = valid['training_features'].tolist()
    x = np.asarray(x)
    dates = valid['Date'].tolist()
    real_x = valid[ticker + LABEL_TO_PREDICT].values
    valid_sample = {'x': x, 'y': y, 'dates': dates, 'p_dates': dates, "real_x": real_x}

    # shuffles dataset
    new_df = df[:-sample_size]
    new_df = new_df.sample(frac=1).reset_index(drop=True)

    # Todo: Balance data
    y = new_df[ticker + LABEL_TO_PREDICT + '_labels'].values
    values, count = np.unique(y, return_counts=True)
    limit = np.min(count)
    limit_value = values[np.argmin(count)]

    # load dataset
    y = new_df[ticker + LABEL_TO_PREDICT + '_labels'].values
    x = new_df['training_features'].tolist()
    x = np.asarray(x)
    dates = new_df['Date'].tolist()

    # clean data
    mask = ~np.isnan(x).any(axis=1)
    x = x[mask];
    y = y[mask]
    mask = np.isfinite(x).any(axis=1)
    x = x[mask];
    y = y[mask]

    print('input training labels shape:', y.shape, ' and features shape:', x.shape)
    return x, y, dates, valid_sample


def train(x, y):
    logging.debug("X sample: \ {} ".format(len(x.shape)))
    logging.debug("y sample: \ {} ".format(len(y.shape)))

    # random shuffle and split
    test_size = int(len(y) * 0.2)
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
    return confidence, clf


def forecast(ticker, df):
    # load dataset
    x, y, dates, valid_set = get_training_dataset(ticker, df)

    # load model
    if not os.path.exists(ticker):

        # train model
        confidence, clf = train(x, y)

        # save model
        if not os.path.exists(MODELS_PATH):
            os.makedirs(MODELS_PATH)
        joblib.dump(clf, MODELS_PATH + ticker + '.pkl')

    # run valid_set
    from src.visualization import matplot_graphs
    valid_set['y_pred'] = clf.predict(valid_set['x'])
    matplot_graphs.plot_histogram(valid_set, ticker, confidence, PREDICTION_DAYS)

    # last dataset sample
    clf = joblib.load(MODELS_PATH + ticker + '.pkl')

    # get forecast
    x = x[-1:]
    forecast_ = clf.predict(x)
    print('\n', ticker, ': Based in the last', BATCH_LEN, 'market days, the forecast is ',
          forecast_ * PERCENTAGE_THRESHOLD, '% in next', PREDICTION_DAYS, 'days\n')


if __name__ == "__main__":
    import warnings, sys

    if not sys.warnoptions:
        warnings.simplefilter("ignore")

    # parser initial arguments
    # logging.basicConfig(filename='log.log', level=logging.INFO)
    parser = argparse.ArgumentParser(description='Optional app description')
    # run -v for debug mode
    parser.add_argument('-v', "--verbose", action='store_true', help='Type -v to do debugging')
    args = parser.parse_args()

    # set debug mode
    if args.verbose: logger.setLevel(logging.DEBUG)

    # forecast chosen ticker
    df = get_dataframe(STOCK_TO_PREDICT)
    forecast(STOCK_TO_PREDICT, df)

    # forecast all sp500
    # sp500_list = getsp500()
    # for sp500_ticker in sp500_list:
    #     df = get_dataframe(sp500_ticker)
    #     forecast(sp500_ticker, df)
