import warnings
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

# %matplotlib inline
import util
from topic_classification.ExperimentController import ExperimentController
from topic_classification.constants import *
from topic_classification.dataset_utils import \
    load_preprocessed_bbc_news_summary, \
    get_dataset_avg_length
from topic_classification.datastructures import TrainingData
from topic_classification.display_utils import \
    get_train_test_distribution_by_labels_names, \
    create_bar_plot, create_2_bar_plot, create_cv_test_time_plots
from topic_classification.feature_extraction_utils import \
    get_simple_bag_of_words_features
from topic_classification.train_utils import train_multiple_classifiers, \
    get_chosen_classifiers

warnings.filterwarnings('ignore')
# Script for different kind of experiments

dataset = Dataset.bbc_news_summary
feature_extraction_method = FeatureExtractionMethod.WORD2VEC
classifiers = [ClassificationMethod.Naive_Bayes_Classifier,
               ClassificationMethod.Logistic_Regression,
               ClassificationMethod.Support_Vector_Machines,
               ClassificationMethod.SVM_with_SGD]

experiment_controller = ExperimentController('tc#2.3', '1')
experiment_controller.run_experiment(dataset, feature_extraction_method,
                                     classifiers, should_load_embedding_model=True)

# classifiers_tuples = (
#     multinominal_naive_bayes_classifier,
#     logistic_regression_classifier,
#     support_vector_machines_classifier,
#     svm_with_stochastic_gradient_descent_classifier,
#     random_forest_classifier,
#     gradient_boosting_machines_classifier
# )
#
# # Load chosen dataset
# data_df = load_preprocessed_bbc_news_summary()
# get_dataset_avg_length(data_df)
#
# # # Split on train and test dataset
# train_corpus, test_corpus, train_label_names, test_label_names = train_test_split(
#     np.array(data_df['Clean Article']),
#     np.array(data_df['Target Name']),
#     test_size=TEST_SET_SIZE_RATIO, random_state=42)
