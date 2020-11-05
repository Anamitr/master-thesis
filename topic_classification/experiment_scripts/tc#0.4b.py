import fasttext
from sklearn.model_selection import train_test_split

import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import importlib

# %matplotlib inline
import util
import text_preprocessing.text_normalizer as tn
from topic_classification.experiment_config import \
    get_chosen_classifiers, CLASSIFIERS_AND_RESULTS_DIR_PATH, \
    RESULTS_PATH, \
    CLASSIFIER_ITERATION, \
    WORD2VEC_MODEL_SAVE_PATH, FAST_TEXT_SAVE_PATH
from topic_classification.constants import *
from topic_classification.dataset_utils import load_20newsgroups, \
    fetch_preprocess_and_save_20newsgroups
from topic_classification.datastructures import TrainingData
from topic_classification.display_utils import \
    create_bar_plot, create_2_bar_plot
from topic_classification.train_utils import train_multiple_classifiers
from topic_classification.feature_extraction_utils import \
    document_vectorize, document_vectorize_with_fasttext_model
from topic_classification.dataset_utils import load_20newsgroups

import util
import topic_classification.experiment_config as experiment_config

from topic_classification.constants import TOPIC_CLASSIFICATION_DATA_PATH

data_df = load_20newsgroups()

train_corpus, test_corpus, train_label_names, \
test_label_names = train_test_split(np.array(data_df['Clean Article']),
                                    np.array(data_df['Target Name']),
                                    test_size=0.33, random_state=42)

# Reformat and save data for FastText
ft_train_data_formatted = ''
for i in range(0, len(train_corpus)):
    if i in data_df.index:
        ft_train_data_formatted += '__label__' + train_label_names[i] + ' ' + \
                                   train_corpus[i] + '\n'
util.save_object(ft_train_data_formatted, TOPIC_CLASSIFICATION_DATA_PATH +
                 DATASET_NAME_20newsgroups + '_fasttext_train_formatted.txt')

ft_test_data_formatted = ''
for i in range(0, len(test_corpus)):
    if i in data_df.index:
        ft_test_data_formatted += '__label__' + test_label_names[i] + ' ' + \
                                  test_corpus[i] + '\n'
util.save_object(ft_test_data_formatted, TOPIC_CLASSIFICATION_DATA_PATH +
                 DATASET_NAME_20newsgroups + '_fasttext_test_formatted.txt')

# Load data for FastText
# ft_data_formatted = util.load_object(TOPIC_CLASSIFICATION_DATA_PATH +
#                                      DATASET_NAME_20newsgroups +
#                                      '_fasttext_formatted.txt')

# Train FastText
fasttext_model = fasttext.train_supervised(TOPIC_CLASSIFICATION_DATA_PATH +
                                           DATASET_NAME_20newsgroups +
                                           '_fasttext_train_formatted.txt',
                                           epoch=500)
fasttext_model.test(TOPIC_CLASSIFICATION_DATA_PATH +
                    DATASET_NAME_20newsgroups +
                    '_fasttext_test_formatted.txt')

predicted_labels = []
num_of_correctly_predicted_labels = 0
for i in range(0, len(test_corpus)):
    predicted_label = fasttext_model.predict(test_corpus[i])
    predicted_label = predicted_label[0][0].replace('__label__', '')
    predicted_labels.append(predicted_label)
    if predicted_label == test_label_names[i]:
        num_of_correctly_predicted_labels += 1
acc = float(num_of_correctly_predicted_labels) / len(test_corpus)
print('acc =', acc)
# acc = 0.7059602649006622
