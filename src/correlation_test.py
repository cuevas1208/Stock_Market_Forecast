import numpy as np
from matplotlib import pyplot as plt
from src.sklearn_main import process_data_for_labels
from src.data_functions.data_load import get_dataframe
from src.conf import LABEL_TO_PREDICT, CORRELATIONS_DAYS, STOCKS_TO_WATCH, BATCH_LEN
from src.visualization.matplot_graphs import plot_stock
import cv2
import pandas as pd

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


def plot_correlation(stock_to_predict):
    set_len = BATCH_LEN

    # load dataset
    df = get_dataframe(stock_to_predict)

    # roll the desired dataset
    stock_vector = df[stock_to_predict+LABEL_TO_PREDICT].shift(-CORRELATIONS_DAYS)

    # time frame correlation
    top_samples = 2
    tmp_df, tmp_stock = df[-set_len:-1].pct_change(), stock_vector[-set_len:-1].pct_change()
    correlation = tmp_df.corrwith(tmp_stock, axis=0)
    top3 = correlation.sort_values(ascending=False)[:top_samples]
    bottom3 = correlation.sort_values(ascending=True)[:top_samples]
    corr_stocks = pd.concat((top3, bottom3))

    # populate buy sell based on the correlation
    set_len += 6
    tmp_df, tmp_stock = df[-set_len:], df[stock_to_predict+LABEL_TO_PREDICT][-set_len:]
    top_results = [[]]*top_samples*2
    bottom_results = [[]]*top_samples*2
    for e, idx in enumerate(corr_stocks.index):
        top_results[e], buy_sell = plot_stock(idx, corr_stocks[idx], tmp_df[idx])
        buy_sell[0] = buy_sell[0] + 1
        buy_sell[1] = buy_sell[1] + 1
        bottom_results[e], _ = plot_stock(stock_to_predict + LABEL_TO_PREDICT, corr_stocks[idx], tmp_stock, buy_sell)
    top_results = np.concatenate(top_results, 1).astype(np.uint8)
    bottom_results = np.concatenate(bottom_results, 1).astype(np.uint8)
    results = np.concatenate((top_results, bottom_results), 0).astype(np.uint8)
    cv2.imshow('correlation', results)
    cv2.waitKeyEx(10000)


if __name__ == "__main__":
    for i in STOCKS_TO_WATCH:
        plot_correlation(i)
