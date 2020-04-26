# constants
DATA_LOC = "data/"
MODELS_PATH = 'models/'
FULL_DATA_PKL = DATA_LOC+ "full_dataframe.pkl"
STOCK_INDEX_LOC = DATA_LOC + "_sp500_list" + '.csv'
BENCHMARK_PKL = DATA_LOC + 'pre_processed/benchmark.pkl'
CORR_ANALYSIS_PKL = DATA_LOC + 'pre_processed/correlation_analysis.pkl'

# it's good to include 2000, 2008 crash in data traing
START_TIME = '1998-01-01'

# vector given at a given time to the model to do a prediction
BATCH_LEN = 10
VISUALIZE_CORRELATION = True
PREDICTION_DAYS = 3  # one day would be prediction for tomorrow, 2 days would be prediction for 2 days from now
STOCK_TO_PREDICT = 'SPY'
LABEL_TO_PREDICT = '_Close'
CORRELATIONS_DAYS = 1  # offset for the correlation

# PERCENTAGE_THRESHOLD should change based on the volatility of the stock
# for example if you want to know if stock would go over or under 2%  # 0.02 == 2 percentage
PERCENTAGE_THRESHOLD = 0.01


STOCKS_TO_WATCH = [
    'QQQ',  # tech
    'SPY',  # top500
    'IWM',  # small cap, russell 200
    'DJT',  # transport
    'SMH',  # semiconductor
    'FAS',  # finance
    'SPXS', 'TZA', 'SOXS',  # bear market index

    # stock on the move cruse
    'CCL', 'NCLH', 'SIX', 'LRN', 'ERI', 'RCL',

    # real state
    'MFA', 'RWT', 'AIM'
]
