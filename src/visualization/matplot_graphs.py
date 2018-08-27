import matplotlib.pyplot as plt
import numpy as np


def plot_histogram(p, t, dates, name='stock', confidence='???', forecast='???'):
    fig, ax = plt.subplots()

    ax.plot(p, C='g', label='prediction')
    ax.plot(t, C='b', label='ground truth')
    ax.legend()

    # window dimensions
    plt.axis([0, len(dates)/2, np.min(p)-1, np.max(t)+1])

    # x labels
    plt.xticks(range(len(dates)), dates)

    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')  # works fine on Windows!

    plt.xlabel('forecast for ' + str(forecast) + ' days from the above date')
    plt.ylabel('buy(1), hold(0), sell(-1)')
    plt.title(name + ' model confidence ' + str(confidence))

    plt.show()
