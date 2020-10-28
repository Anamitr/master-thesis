import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import importlib

# %matplotlib inline
import util
import text_preprocessing.text_normalizer as tn
from topic_classification.experiment_config import \
    get_basic_statistical_classifiers, CLASSIFIERS_SAVE_PATH, RESULTS_PATH, \
    CLASSIFIER_ITERATION, \
    WORD2VEC_MODEL_SAVE_PATH
from topic_classification.constants import *
from topic_classification.dataset_utils import load_20newsgroups, \
    fetch_preprocess_and_save_20newsgroups, load_preprocessed_news_category_dataset
from topic_classification.datastructures import TrainingData
from topic_classification.display_utils import \
    create_bar_plot, create_2_bar_plot
from topic_classification.train_utils import train_multiple_classifiers
from topic_classification.feature_extraction_utils import \
    document_vectorize

import gensim
import logging  # Setting up the loggings to monitor gensim

logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s",
                    datefmt='%H:%M:%S', level=logging.INFO)

warnings.filterwarnings('ignore')

# # Fetch and preprocess data or load from disk
# data_df = fetch_preprocess_and_save_news_category_dataset()

# # Load dataset from disk
data_df = load_preprocessed_news_category_dataset()

train_corpus, test_corpus, train_label_names, test_label_names = train_test_split(
    np.array(data_df['Clean Article']),
    np.array(data_df['Target Name']),
    test_size=0.33, random_state=42)
# tokenize corpus
tokenized_train = [tn.tokenizer.tokenize(text) for text in train_corpus]
tokenized_test = [tn.tokenizer.tokenize(text) for text in test_corpus]
# generate word2vec word embeddings


# # build and save word2vec model
w2v_num_features = 1000
# w2v_model = gensim.models.Word2Vec(sentences=tokenized_train, size=w2v_num_features,
#                                    window=100, min_count=2, sample=1e-3, sg=1,
#                                    iter=5, workers=10)
# util.save_object(w2v_model, CLASSIFIERS_SAVE_PATH + 'w2v_model' + str(
#     CLASSIFIER_ITERATION) + '.pkl')
# # Load word2vec model
w2v_model = util.load_object(WORD2VEC_MODEL_SAVE_PATH)

# generate document level embeddings
# remember we only use train dataset vocabulary embeddings
# so that test dataset truly remains an unseen dataset
# generate averaged word vector features from word2vec model
avg_wv_train_features = document_vectorize(corpus=tokenized_train,
                                           model=w2v_model,
                                           num_features=w2v_num_features)
avg_wv_test_features = document_vectorize(corpus=tokenized_test,
                                          model=w2v_model,
                                          num_features=w2v_num_features)
print('Word2Vec model:> Train features shape:', avg_wv_train_features.
      shape, ' Test features shape:', avg_wv_test_features.shape)

# # pack data in one class
training_data = TrainingData(avg_wv_train_features, train_label_names,
                             avg_wv_test_features, test_label_names)

# # Get classifier definitions
classifier_list, classifier_name_list, classifier_name_shortcut_list = \
    get_basic_statistical_classifiers()


def train_and_save(classifier_list, classifier_name_list, training_data):
    results = train_multiple_classifiers(classifier_list, classifier_name_list,
                                         training_data)
    util.save_object(results, RESULTS_PATH)
    util.save_classifier_list(classifier_list, classifier_name_list,
                              CLASSIFIERS_SAVE_PATH)
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
