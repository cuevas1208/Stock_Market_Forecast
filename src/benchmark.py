import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve, accuracy_score
from tqdm import tqdm
import os

from src.conf import BATCH_LEN, CORR_ANALYSIS_PKL, BENCHMARK_PKL
from src.correlation_test import correlations
from src.data_functions.data_load import get_dataframe


def generator(names, post_list, date_range):

    # iterate by the last 30 days or random from a range of dates
    for stock_name in tqdm(names):

        # slice the dataframe to random date
        for post_fix in post_list:

            for i in range(1, date_range):
                yield i, stock_name+post_fix, post_fix


def benchmark():
    """ end goal is to know the accuracy of the method with an optimal general and local threshold
    :return: probability """
    post_list = ['_Close', '_High', '_Low']

    # iterate stock
    df = get_dataframe()
    indexes = df.columns

    # iterate index (_Low, _High and _Close)
    names = [index.split("_")[0] for index in indexes]
    names = np.unique(names)

    # run prediction
    our_generator = generator(names, post_list, date_range=30)
    corr = correlations(best_match=True, top_samples=2, set_len=BATCH_LEN)

    results_df = pd.DataFrame()
    for e, (i, stock_name, post_fix) in enumerate(our_generator):
        if stock_name not in indexes:
            continue
        gt_df = df[:-i+1][stock_name] if i>1 else df[stock_name]
        gt = gt_df.values[-1] - gt_df.values[-2]
        if gt > 0: gt = 1
        if gt < 0: gt = -1
        _, n_best_corr, n_inv_best_corr = corr.get_correlation(df[:-i], stock_name)
        pred, prob = corr.predict_correlation(df[:-i], stock_name, visualize=False)
        row = {'stock_name':stock_name, 'gt':gt, 'm_pred':np.mean(pred), 'm_prob':np.mean(prob),
               'pred':pred, 'prob':prob, 'n_best_corr':n_best_corr, 'n_inv_best_corr':n_inv_best_corr,
               'post_index': post_fix}
        results_df = results_df.append(row, ignore_index=True)

        if not e % 30:
            results_df.to_pickle(BENCHMARK_PKL)
    results_df.to_pickle(BENCHMARK_PKL)
    return None


def eer_funciton(y, y_score):
    """ y is True binary labels in range {0, 1} or {-1, 1}.
        y_score is Target scores, can either be probability, confidence, or
    :return thresh:
    """
    if len(y) < 2:
        return 0, 0
    fpr, tpr, thresholds = roc_curve(y, y_score, pos_label=1)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return thresh, eer


def get_accuracy_score(results_df):

    # compares it with ground truth, stores results and probability in a pickle file
    gt, pred, prob = results_df['gt'].values, results_df['m_pred'].values, results_df['m_prob'].values

    # adds increase if you pick pred at 1 or -1
    # pred[pred < 0] = -1
    # pred[pred > 0] = 1
    filter = np.logical_or(pred == 1, pred == -1)
    y_score = np.absolute(prob[filter])
    y_true, y_pred = pred[filter], gt[filter]

    # general score
    # no threshold round
    accuracy_list = []
    accuracy = accuracy_score(y_true, y_pred)
    accuracy_list.append((accuracy, 0, len(y_pred)))
    if accuracy == 0 or accuracy == 1:
        return accuracy_list[0]

    # with threshold round
    y = np.equal(y_true, y_pred)
    thresh, eer = eer_funciton(y, y_score)
    if max(y_score) <= thresh:
        return accuracy_list[0]
    filter = y_score > thresh
    accuracy = accuracy_score(y_true[filter], y_pred[filter])
    accuracy_list.append((accuracy, thresh, len(y_pred[filter])))
    if accuracy == 1:
        return accuracy_list[1]

    # 2nd threshold
    if np.any(y[filter]):
        thresh, eer = eer_funciton(y[filter], y_score[filter])
        if max(y_score) > thresh:
            filter = y_score > thresh
            accuracy = accuracy_score(y_true[filter], y_pred[filter])
            accuracy_list.append((accuracy, thresh, len(y_pred[filter])))

    arg = np.argmax(accuracy_list, axis=0)[0]
    return accuracy_list[arg]


def analice_results():
    # store results
    corr_analysis = pd.DataFrame()

    # load results
    print(os.path.isfile(BENCHMARK_PKL))
    results_df = pd.read_pickle(BENCHMARK_PKL)
    results_df.dropna(inplace=True, axis=0)

    # pick general probability, run another random test
    get_accuracy_score(results_df)

    # get results test general probability and by stock probability
    names = np.unique(results_df['stock_name'].values)
    for stocks_name in names:
        df_by_name = results_df[results_df['stock_name'] == stocks_name]
        score, threshold, sample_size = get_accuracy_score(df_by_name)
        row = {'stock_name': stocks_name, 'accuracy': score, 'threshold':threshold, 'sample_size':sample_size}
        corr_analysis = corr_analysis.append(row, ignore_index=True)
        if score > .8 and sample_size > 1:
            print(stocks_name, score, threshold, sample_size)

    corr_analysis.sort_values(by=['sample_size', 'accuracy'], inplace=True, ascending=False)
    corr_analysis.to_pickle(CORR_ANALYSIS_PKL)

    return None


if __name__ == "__main__":
    # benchmark()
    analice_results()
