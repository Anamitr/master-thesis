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
    CLASSIFIERS_AND_RESULTS_DIR_PATH, \
    RESULTS_PATH, \
    CLASSIFIER_ITERATION, \
    WORD2VEC_MODEL_SAVE_PATH, FAST_TEXT_SAVE_PATH
from topic_classification.constants import *
from topic_classification.dataset_utils import load_20newsgroups, \
    fetch_preprocess_and_save_20newsgroups
from topic_classification.datastructures import TrainingData
from topic_classification.display_utils import \
    create_bar_plot, create_2_bar_plot
from topic_classification.train_utils import train_multiple_classifiers, \
    get_chosen_classifiers
from topic_classification.feature_extraction_utils import \
    document_vectorize, document_vectorize_with_fasttext_model
from topic_classification.dataset_utils import load_20newsgroups

import util
import topic_classification.experiment_config as experiment_config

from topic_classification.constants import TOPIC_CLASSIFICATION_DATA_PATH
import tensorflow_hub as hub
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

data_df = load_20newsgroups()
data_string = util.load_object(TOPIC_CLASSIFICATION_DATA_PATH +
                               '20_newsgroups_one_string.txt')
data_word_list = data_string.split(' ')
vocabulary = set(data_word_list)

train_corpus, test_corpus, train_label_names, \
test_label_names = train_test_split(np.array(data_df['Clean Article']),
                                    np.array(data_df['Target Name']),
                                    test_size=0.33, random_state=42)
# tokenize corpus
tokenized_train = [tn.tokenizer.tokenize(text) for text in train_corpus]
tokenized_test = [tn.tokenizer.tokenize(text) for text in test_corpus]

# # # Feature extraction
# elmo = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)
elmo = hub.KerasLayer("https://hub.tensorflow.google.cn/google/elmo/3",
                      trainable=True, output_key='elmo')


def elmo_vectors(x):
    embeddings = elmo(x, signature="default", as_dict=True)["elmo"]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        # return average of ELMo features
        return sess.run(tf.reduce_mean(embeddings, 1))


SENTENCES_BATCH_SIZE = 100
list_train = [tokenized_train[i:i + SENTENCES_BATCH_SIZE] for i in
              range(0, len(tokenized_train), SENTENCES_BATCH_SIZE)]
list_test = [tokenized_test[i:i + SENTENCES_BATCH_SIZE] for i in
             range(0, len(tokenized_test), SENTENCES_BATCH_SIZE)]

# Extract ELMo embeddings
elmo_train = [elmo_vectors(x) for x in tokenized_train]
elmo_test = [elmo_vectors(x) for x in tokenized_test]

# training_data = TrainingData(avg_ft_train_features, train_label_names,
#                              avg_ft_test_features, test_label_names)
#
# classifier_list, classifier_name_list, classifier_name_shortcut_list = \
#     get_chosen_classifiers()
#
#
# def train_and_save(classifier_list, classifier_name_list, training_data):
#     results = train_multiple_classifiers(classifier_list, classifier_name_list,
#                                          training_data)
#     util.save_object(results, RESULTS_PATH)
#     util.save_classifier_list(classifier_list, classifier_name_list,
#                               CLASSIFIERS_AND_RESULTS_DIR_PATH)
#     return results
#
#
# results = train_and_save(classifier_list, classifier_name_list, training_data)
#
# mean_scores = [round(result[1], SCORE_DECIMAL_PLACES) for result in results]
# test_scores = [round(result[2], SCORE_DECIMAL_PLACES) for result in results]
# elapsed_times = [round(result[3], TIME_DECIMAL_PLACES) for result in results]
#
# create_2_bar_plot(classifier_name_shortcut_list, 'Classifier scores', 'Accuracy',
#                   mean_scores, test_scores, 'cv means', 'test set',
#                   y_range_tuple=(0, 1))
# create_bar_plot(classifier_name_shortcut_list, 'Elapsed training times',
#                 'Time in seconds', elapsed_times, color='red')
