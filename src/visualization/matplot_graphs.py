import matplotlib.pyplot as plt
import numpy as np
import cv2

def plot_histogram(df, name='stock', confidence='???', forecast='???'):
    pred = df['y_pred']
    dates = df['dates']
    truth = df['y']

    fig, ax = plt.subplots()
    ax.plot(pred, C='g', label='prediction')
    ax.plot(truth, C='b', label='ground truth')
    ax.legend()

    # window dimensions
    plt.subplots(2, 1, 'row')
    plt.axis([0, len(dates)/2, np.min(pred)-1, np.max(truth)+1])
    plt.xlabel('forecast for ' + str(forecast) + ' days from the above date')
    plt.ylabel('buy(1), hold(0), sell(-1)')
    plt.title(name + ' model confidence ' + str(confidence))
    states_buy = np.argwhere(pred == 1).flatten().tolist()
    states_sell = np.argwhere(pred == -1).flatten().tolist()
    dataset = df['real_x']

    plt.subplot(2, 1, 2)
    plt.figure(figsize=(20, 10))
    plt.plot(dataset, label='true close', c='g')
    plt.plot(dataset, 'X', label='predict buy', markevery=states_buy, c='b')
    plt.plot(dataset, 'o', label='predict sell', markevery=states_sell, c='r')
    plt.legend()
    plt.show()


def plot_stock(name, confidence, dataset, buy_sale = []):
    # Todo plot bottom dates
    if not len(buy_sale):
        roll = np.roll(dataset.values, -1)
        roll -= dataset.values
        states_buy = np.argwhere(roll[:-1]>0).flatten()
        states_sell = np.argwhere(roll[:-1]<0).flatten()
    else:
        if confidence > 0:
            states_buy, states_sell = buy_sale
        elif confidence < 0:
            states_sell, states_buy = buy_sale

    plt.title(name + ' - ' + str(confidence))
    plt.plot(dataset, 'X', label='predict buy', markevery=states_buy.tolist(), c='b')
    plt.plot(dataset, 'o', label='predict sell', markevery=states_sell.tolist(), c='r')
    plt.plot(dataset, label=str(confidence), c='g')
    plt.legend()
    plt.savefig('tmp.png')
    plt.close()
    return cv2.imread('tmp.png'), [states_buy, states_sell]
