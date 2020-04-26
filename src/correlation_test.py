import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os

from src.conf import CORRELATIONS_DAYS, BATCH_LEN, STOCKS_TO_WATCH, \
    LABEL_TO_PREDICT, CORR_ANALYSIS_PKL
from src.visualization.matplot_graphs import plot_stock
from src.data_functions.data_load import get_dataframe


def detaCorrelation(df, stocksName):
    """ graphs data correlation
        do not run this functions with a large dataframe it would take for ever
    """
    df_corr = df.corr()

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


class correlations():

    def __init__(self, best_match, top_samples=2, set_len=BATCH_LEN):
        # 2 pluss, in the process of correlation 2 values would be lost
        self.set_len = set_len+2
        self.best_match = best_match
        self.n_samples = top_samples
        self.corr_stocks = None
        self.pred = None

    def get_correlation(self, df_in, stock_to_predict):

        # roll the desired dataset
        df = df_in.copy()
        stock_vector = df[stock_to_predict].shift(-CORRELATIONS_DAYS)

        # filter correct moves
        if self.best_match:
            self.pred, tmp_stock0 = df[-self.set_len:].pct_change()[1:], stock_vector[-self.set_len:-1].pct_change()[1:]
            # drop any invalid column
            self.pred.dropna(inplace=True, axis=1)
            self.pred[self.pred < 0] = -1
            self.pred[self.pred > 0] = 1
            tmp_stock0[tmp_stock0 < 0] = -1
            tmp_stock0[tmp_stock0 > 0] = 1
            # subtract and absolute value
            sub = self.pred[:-1].sub(tmp_stock0, axis=0)
            sub[sub < 0] = sub[sub < 0]*-1
            top_match = sub.sum(axis=0)
            top_match = top_match.sort_values(ascending=True)
            min_matches = int(top_match[self.n_samples-1])
            top = top_match[top_match <= min_matches]
            dataset = df[top.index]
            n_best_corr = len(top.index)
            # print('ground truth:\n', tmp_stock0)
            # print(len(top.index), ' best match:\n', self.pred[top.index])
        else:
            dataset = df

        # make data correlation
        tmp_df, tmp_stock = dataset[-self.set_len:-1].pct_change()[1:], stock_vector[-self.set_len:-1].pct_change()[1:]
        correlation = tmp_df.corrwith(tmp_stock, axis=0)
        # correlation[top.index[0]] += 1
        top3 = correlation.sort_values(ascending=False)[:self.n_samples]
        # top3[0] -= 1

        # filter worst correlation
        if self.best_match:
            top_match = top_match.sort_values(ascending=False)
            min_matches = int(top_match[self.n_samples-1])
            top = top_match[top_match >= min_matches]
            n_inv_best_corr = len(top.index)
            # print(len(top.index), ' inverse match:\n', self.pred[top.index])
            dataset = df[top.index]
            tmp_df = dataset[-self.set_len:-1].pct_change()[1:]
            tmp_df.dropna(inplace=True, axis=1)
            correlation = tmp_df.corrwith(tmp_stock, axis=0)
            # correlation[top.index[0]] -= 1
        bottom3 = correlation.sort_values(ascending=True)[:self.n_samples]
        # bottom3[0] += 1
        self.corr_stocks = pd.concat((top3, bottom3))

        return self.corr_stocks, n_best_corr, n_inv_best_corr

    def predict_correlation(self, df, stock_to_predict, threshold=0.5, visualize=True):

        # populate buy sell based on the correlation
        set_len = self.set_len + 6
        tmp_df, tmp_stock = df[-set_len:], df[stock_to_predict][-set_len:]
        top_results = [[]]*len(self.corr_stocks)
        bottom_results = [[]]*len(self.corr_stocks)

        results = []
        for e, idx in enumerate(self.corr_stocks.index):
            prediction = self.pred[idx].values[-1]
            prediction = prediction if self.corr_stocks[idx] > 0 else prediction * -1
            probability = abs(self.corr_stocks[idx]) * prediction
            results.append((prediction, probability))
            if visualize:
                top_results[e], [buy, sell] = plot_stock(idx, self.corr_stocks[idx], tmp_df[idx])
                bottom_results[e], _ = plot_stock(stock_to_predict, self.corr_stocks[idx], tmp_stock, [buy+1, sell+1])

        # concatenate image for view
        prediction, probability = np.array(results).T
        prediction, probability = np.mean(prediction), np.abs(np.mean(probability))

        if probability > threshold and prediction != 0:
            print("{} prediction {} prob {}".format(stock_to_predict, prediction, probability))

            if visualize:
                top_results = np.concatenate(top_results, 1).astype(np.uint8)
                bottom_results = np.concatenate(bottom_results, 1).astype(np.uint8)
                ouput_img = np.concatenate((top_results, bottom_results), 0).astype(np.uint8)
                cv2.imshow('correlation', ouput_img)
                cv2.waitKeyEx()

        return prediction, probability


if __name__ == "__main__":

    # get accuracy above
    print(os.getcwd())
    ca_df = pd.read_pickle(CORR_ANALYSIS_PKL)
    ca_df = ca_df[ca_df['accuracy'].values > 0.8]
    ca_df = ca_df[ca_df['sample_size'].values > 2]

    corr = correlations(best_match=True, top_samples=2, set_len=BATCH_LEN)

    for index, row in ca_df.iterrows():
        # load dataset
        stock_name = row['stock_name']
        threshold = row['threshold']

        df = get_dataframe()
        corr.get_correlation(df, stock_name)
        corr.predict_correlation(df, stock_name, threshold, visualize=False)

    # for i in STOCKS_TO_WATCH:
    #     # load dataset
    #     df = get_dataframe(i)
    #     corr.get_correlation(df, i+LABEL_TO_PREDICT)
    #     corr.plot_correlation(df, i+LABEL_TO_PREDICT, visualize=True)

# Todo: Output should be a dataframe of:
#  stock name, correlation score, correlation 1, correlation score, correlation 2..

# Todo make a long term test to find how good this tool is
