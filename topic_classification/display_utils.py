from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import util

from topic_classification.constants import *


def get_train_test_distribution_by_labels_names(train_label_names, test_label_names):
    trd = dict(Counter(train_label_names))
    tsd = dict(Counter(test_label_names))
    return (pd.DataFrame([[key, trd[key], tsd[key]] for key in trd],
                         columns=['Target Label', 'Train Count', 'Test Count'])
            .sort_values(by=['Train Count', 'Test Count'],
                         ascending=False))


def create_bar_plot(classifier_name_list, scores, plot_title, y_label_name,
                    color='blue'):
    x = np.arange(len(classifier_name_list))
    fig, ax = plt.subplots()
    bars = ax.bar(x, scores, BAR_WIDTH, color=color)

    ax.set_ylabel(plot_title)
    ax.set_title(y_label_name)
    ax.set_xticks(x)
    ax.set_xticklabels(classifier_name_list)

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(bars)
    # fig.tight_layout()
    plt.show()
    fig.savefig(
        SAVE_PATH + util.convert_name_to_filename(plot_title) + '_' + str(
            CLASSIFIER_ITERATION) + '.png')


def display_bar_plot(title, labels, scores, y_label):
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_title(title)
    ax.set_ylabel(y_label)
    plt.ylabel = y_label
    ax.bar(labels, scores)
    plt.show()
    plt.savefig(SAVE_PATH + 'plot.png')
