import sys

from sklearn.datasets import fetch_20newsgroups
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import importlib

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB

# %matplotlib inline
import util
from topic_classification.classifire_definiton_utils import \
    get_basic_statistical_classifiers
from topic_classification.constants import *
import topic_classification.display_utils as display_utils
from topic_classification.train_utils import train_multiple_classifiers
from topic_classification.constants import BAR_WIDTH
from topic_classification.dataset_utils import load_20newsgroups
from topic_classification.datastructures import TrainingData
from topic_classification.display_utils import \
    get_train_test_distribution_by_labels_names, \
    plot_classifiers_scores_and_training_time_as_bars

warnings.filterwarnings('ignore')

# Fetch and preprocess data or load from disk
# data_df = fetch_preprocess_and_save_20newsgroups()

data_df = load_20newsgroups()

# For some reason dataset still contains np.nan
data_df = data_df.dropna()

# Split on train and test dataset
train_corpus, test_corpus, train_label_nums, test_label_nums, train_label_names, \
test_label_names = train_test_split(np.array(data_df['Clean Article']),
                                    np.array(data_df['Target Label']),
                                    np.array(data_df['Target Name']),
                                    test_size=0.33, random_state=42)

# Table of category, train count, test count
train_test_distribution = get_train_test_distribution_by_labels_names(train_label_names,
                                                                      test_label_names)

# build BOW features on train articles
cv = CountVectorizer(binary=False, min_df=0.0, max_df=1.0)
cv_train_features = cv.fit_transform(train_corpus)
# transform test articles into features
cv_test_features = cv.transform(test_corpus)
print('BOW model:> Train features shape:', cv_train_features.shape,
      ' Test features shape:', cv_test_features.shape, '\n')
# pack data in one class
training_data = TrainingData(cv_train_features, train_label_names,
                             cv_test_features, test_label_names)
# Get classifier definitions
classifier_list, classifier_name_list, classifier_name_shortcut_list = \
    get_basic_statistical_classifiers()


def train_and_save(classifier_list, classifier_name_list, training_data):
    results = train_multiple_classifiers(classifier_list, classifier_name_list,
                                         training_data)
    util.save_object(results, RESULTS_PATH)
    util.save_classifier_list(classifier_list, classifier_name_list, SAVE_PATH)
    return results


# results = train_and_save(classifier_list, classifier_name_list, training_data)
# Load from disk
results = util.load_object(RESULTS_PATH)
classifier_list = util.load_classifier_list(classifier_name_list, SAVE_PATH)

# results[0] = array of crossvalidation, [1] crossvalidation scores,
# [2] test score, [3] times
scores = [round(result[1], 2) for result in results]
times = [round(result[3], 2) for result in results]
plot_classifiers_scores_and_training_time_as_bars(classifier_name_shortcut_list, scores)
