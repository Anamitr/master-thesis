from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import util

from topic_classification.constants import *
from topic_classification.experiment_config import CLASSIFIER_ITERATION, \
    CLASSIFIERS_SAVE_PATH


def get_train_test_distribution_by_labels_names(train_label_names, test_label_names):
    trd = dict(Counter(train_label_names))
    tsd = dict(Counter(test_label_names))
    return (pd.DataFrame([[key, trd[key], tsd[key]] for key in trd],
                         columns=['Target Label', 'Train Count', 'Test Count'])
            .sort_values(by=['Train Count', 'Test Count'],
                         ascending=False))


def create_bar_plot(classifier_name_shortcut_list, plot_title, y_label_name, scores,
                    y_range_tuple=None, color='blue'):
    x = np.arange(len(classifier_name_shortcut_list))
    fig, ax = plt.subplots()
    bars = ax.bar(x, scores, BAR_WIDTH, color=color)

    ax.set_ylabel(y_label_name)
    ax.set_title(plot_title)
    ax.set_xticks(x)
    ax.set_xticklabels(classifier_name_shortcut_list)

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
    if y_range_tuple is not None:
        plt.ylim((0, 1))
    plt.show()
    fig.savefig(
        CLASSIFIERS_SAVE_PATH + util.convert_name_to_filename(plot_title) + '_'
        + str(CLASSIFIER_ITERATION) + '.png')


def create_2_bar_plot(classifier_name_shortcut_list, plot_title, y_label_name,
                      scores1, scores2, scores1_label, scores2_label,
                      y_range_tuple=None, color1='blue', color2='orange'):
    x = np.arange(len(classifier_name_shortcut_list))
    fig, ax = plt.subplots()
    bars1 = ax.bar(x - BAR_WIDTH / 2, scores1, BAR_WIDTH, color=color1,
                   label=scores1_label)
    bars2 = ax.bar(x + BAR_WIDTH / 2, scores2, BAR_WIDTH, color=color2,
                   label=scores2_label)

    ax.set_ylabel(y_label_name)
    ax.set_title(plot_title)
    ax.set_xticks(x)
    ax.set_xticklabels(classifier_name_shortcut_list)
    ax.legend()

    bar_labels_y_list = []

    def autolabel(rects, bar_labels_y_list: list, bars_num=1):
        """Attach a text label above each bar in *rects*, displaying its height."""
        y_list = []
        for i in range(0, len(rects)):
            rect = rects[i]
            height = rect.get_height()
            y = height
            y_offset = 3
            if bars_num == 1:
                y_list.append(y)
            elif abs(bar_labels_y_list[bars_num - 2][i] - y) < 0.03:
                y_offset += 10
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, y),
                        xytext=(0, y_offset),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
            bar_labels_y_list.append(y_list)

    autolabel(bars1, bar_labels_y_list)
    autolabel(bars2, bar_labels_y_list, 2)
    # fig.tight_layout()
    if y_range_tuple is not None:
        plt.ylim((0, 1))
    plt.show()
    fig.savefig(
        CLASSIFIERS_SAVE_PATH + util.convert_name_to_filename(plot_title) + '_'
        + str(CLASSIFIER_ITERATION) + '.png')


def display_bar_plot(title, labels, scores, y_label):
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_title(title)
    ax.set_ylabel(y_label)
    plt.ylabel = y_label
    ax.bar(labels, scores)
    plt.show()
    plt.savefig(CLASSIFIERS_SAVE_PATH + 'plot.png')
