import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import importlib

# %matplotlib inline
import util
import text_preprocessing.text_normalizer as tn
from topic_classification.experiment_config import \
    get_basic_statistical_classifiers, CLASSIFIERS_AND_RESULTS_DIR_PATH, RESULTS_PATH, \
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
    document_vectorize

import gensim
from gensim.models.fasttext import FastText
import logging  # Setting up the loggings to monitor gensim

logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s",
                    datefmt='%H:%M:%S', level=logging.INFO)

warnings.filterwarnings('ignore')

# # Fetch and preprocess data or load from disk
# data_df = fetch_preprocess_and_save_20newsgroups()

# # Load dataset from disk
data_df = load_20newsgroups()

train_corpus, test_corpus, train_label_nums, test_label_nums, train_label_names, \
test_label_names = train_test_split(np.array(data_df['Clean Article']),
                                    np.array(data_df['Target Label']),
                                    np.array(data_df['Target Name']),
                                    test_size=0.33, random_state=42)
# tokenize corpus
tokenized_train = [tn.tokenizer.tokenize(text) for text in train_corpus]
tokenized_test = [tn.tokenizer.tokenize(text) for text in test_corpus]

# # # FastText
# # Train and save FastText
ft_num_features = 1000
ft_model = FastText(tokenized_train, size=ft_num_features, window=20,
                    min_count=2, sample=1e-3, sg=1, iter=5, workers=10)
util.save_object(ft_model, FAST_TEXT_SAVE_PATH)
# # Load FastText
# ft_model = util.load_object(FAST_TEXT_SAVE_PATH)

# generate averaged word vector features from word2vec model
avg_ft_train_features = document_vectorize(corpus=tokenized_train,
                                           model=ft_model,
                                           num_features=ft_num_features)
avg_ft_test_features = document_vectorize(corpus=tokenized_test,
                                          model=ft_model,
                                          num_features=ft_num_features)

print('FastText model:> Train features shape:', avg_ft_train_features.shape,
      ' Test features shape:', avg_ft_test_features.shape)

# # pack data in one class
training_data = TrainingData(avg_ft_train_features, train_label_names,
                             avg_ft_test_features, test_label_names)

# # Get classifier definitions
classifier_list, classifier_name_list, classifier_name_shortcut_list = \
    get_basic_statistical_classifiers()


def train_and_save(classifier_list, classifier_name_list, training_data):
    results = train_multiple_classifiers(classifier_list, classifier_name_list,
                                         training_data)
    util.save_object(results, RESULTS_PATH)
    util.save_classifier_list(classifier_list, classifier_name_list,
                              CLASSIFIERS_AND_RESULTS_DIR_PATH)
    return results


# Train and save on disk
results = train_and_save(classifier_list, classifier_name_list, training_data)
# # Load from disk
# classifier_list = util.load_classifier_list(classifier_name_list,
#                                             CLASSIFIERS_SAVE_PATH)
# results = util.load_object(RESULTS_PATH)

# results[0] = array of crossvalidation, [1] crossvalidation scores,
# [2] test score, [3] times
# # Plotting
mean_scores = [round(result[1], SCORE_DECIMAL_PLACES) for result in results]
test_scores = [round(result[2], SCORE_DECIMAL_PLACES) for result in results]
elapsed_times = [round(result[3], TIME_DECIMAL_PLACES) for result in results]
# create_bar_plot(classifier_name_shortcut_list, 'Classifier scores', 'Accuracy',
#                 cv_mean_scores, y_range_tuple=(0, 1))
create_2_bar_plot(classifier_name_shortcut_list, 'Classifier scores', 'Accuracy',
                  mean_scores, test_scores, 'cv means', 'test set',
                  y_range_tuple=(0, 1))
create_bar_plot(classifier_name_shortcut_list, 'Elapsed training times',
                'Time in seconds', elapsed_times, color='red')

# cv_results = util.load_results_of_specific_topic_classification_experiment('tc#0', 1)
# tv_results = results
#
# columns = ['Model', 'CV Score (TF)', 'Test Score (TF)',
#            'CV Score (TF-IDF)', 'Test Score (TF-IDF)']
# cv_tf_idf_comparison = pd.DataFrame(
#     [[classifier_name_list[i], round(cv_results[i][1], 4),
#       round(cv_results[i][2], 4),
#       round(tv_results[i][1], 4), round(tv_results[i][2], 4)]
#      for i in range(0, len(classifier_name_list))],
#     columns=['Model', 'CV Score (TF)',
#              'Test Score (TF)',
#              'CV Score (TF-IDF)',
#              'Test Score (TF-IDF)']).T
# cv_tf_idf_comparison.to_clipboard()
