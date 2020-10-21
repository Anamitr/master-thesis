from collections import Counter
import pandas as pd


def get_train_test_distribution_by_labels_names(train_label_names, test_label_names):
    trd = dict(Counter(train_label_names))
    tsd = dict(Counter(test_label_names))
    return (pd.DataFrame([[key, trd[key], tsd[key]] for key in trd],
                         columns=['Target Label', 'Train Count', 'Test Count'])
            .sort_values(by=['Train Count', 'Test Count'],
                         ascending=False))
