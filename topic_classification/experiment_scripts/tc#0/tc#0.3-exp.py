import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import importlib

# %matplotlib inline
import util
import text_preprocessing.text_normalizer as tn
from topic_classification.experiment_config import \
    CLASSIFIERS_AND_RESULTS_DIR_PATH, RESULTS_PATH, \
    CLASSIFIER_ITERATION, \
    WORD2VEC_MODEL_SAVE_PATH
from topic_classification.constants import *
from topic_classification.dataset_utils import load_20newsgroups, \
    fetch_preprocess_and_save_20newsgroups
from topic_classification.datastructures import TrainingData
from topic_classification.display_utils import \
    create_bar_plot, create_2_bar_plot
from topic_classification.train_utils import train_multiple_classifiers, \
    get_chosen_classifiers
from topic_classification.feature_extraction_utils import \
    document_vectorize

import gensim
import logging  # Setting up the loggings to monitor gensim

logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s",
                    datefmt='%H:%M:%S', level=logging.INFO)

warnings.filterwarnings('ignore')

# # Fetch and preprocess data or load from disk
# data_df = fetch_preprocess_and_save_20newsgroups()

# # Load dataset from disk
data_df = load_20newsgroups()

train_corpus, test_corpus, train_label_names, test_label_names = train_test_split(
    np.array(data_df['Clean Article']),
    np.array(data_df['Target Name']),
    test_size=0.33, random_state=42)

###
# feature engineering with GloVe model
TRAIN_NLP_PATH = CLASSIFIERS_AND_RESULTS_DIR_PATH + 'train_nlp_' + \
                 str(CLASSIFIER_ITERATION) + '.pkl'
TEST_NLP_PATH = CLASSIFIERS_AND_RESULTS_DIR_PATH + 'test_nlp_' + str(CLASSIFIER_ITERATION) + \
                '.pkl'


def read_from_spacy_and_save():
    print('Reading embeddings from spacy (GloVe)')
    print('Train features')
    train_nlp = [tn.spacy_english_model(item) for item in train_corpus]
    util.save_object(train_nlp, CLASSIFIERS_AND_RESULTS_DIR_PATH + 'train_nlp_' +
                     str(CLASSIFIER_ITERATION) + '.pkl')
    train_glove_features = np.array([item.vector for item in train_nlp])
    print('Test features')
    test_nlp = [tn.spacy_english_model(item) for item in test_corpus]
    util.save_object(train_nlp, CLASSIFIERS_AND_RESULTS_DIR_PATH + 'test_nlp_' +
                     str(CLASSIFIER_ITERATION) + '.pkl')
    test_glove_features = np.array([item.vector for item in test_nlp])
    return train_glove_features, test_glove_features


# train_glove_features, test_glove_features = read_from_spacy_and_save()
train_glove_features = util.load_object(TRAIN_NLP_PATH)
test_glove_features = util.load_object(TEST_NLP_PATH)
print('GloVe model:> Train features shape:', train_glove_features.shape,
      ' Test features shape:', test_glove_features.shape)


###

def train_sgd():
    svm = SGDClassifier(loss='hinge', penalty='l2', random_state=42, max_iter=500)
    svm.fit(train_glove_features, train_label_names)
    svm_glove_cv_scores = cross_val_score(svm, train_glove_features,
                                          train_label_names, cv=5)
    svm_glove_cv_mean_score = np.mean(svm_glove_cv_scores)
    print('CV Accuracy (5-fold):', svm_glove_cv_scores)
    print('Mean CV Accuracy:', svm_glove_cv_mean_score)
    svm_glove_test_score = svm.score(test_glove_features, test_label_names)
    print('Test Accuracy:', svm_glove_test_score)


# # pack data in one class
training_data = TrainingData(train_glove_features, train_label_names,
                             test_glove_features, test_label_names)

# # Get classifier definitions
classifier_list, classifier_name_list, classifier_name_shortcut_list = \
    get_chosen_classifiers()


def train_and_save(classifier_list, classifier_name_list, training_data):
    results = train_multiple_classifiers(classifier_list, classifier_name_list,
                                         training_data)
    util.save_object(results, RESULTS_PATH)
    util.save_classifier_list(classifier_list, classifier_name_list,
                              CLASSIFIERS_AND_RESULTS_DIR_PATH)
    return results


# # Train and save on disk
results = train_and_save(classifier_list, classifier_name_list, training_data)
# # Load from disk
# classifier_list = util.load_classifier_list(classifier_name_list,
#                                             CLASSIFIERS_AND_RESULTS_DIR_PATH)
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
