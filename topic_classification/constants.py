from enum import Enum

# Datasets
TOPIC_CLASSIFICATION_DATA_PATH = '/home/konrad/Repositories/master-diploma/' \
                                 'topic_classification/topic_class_data/'
DATASET_NAME_20newsgroups = 'ds20newsgroups'
DATASET_NAME_news_category_dataset = 'news_category_dataset'
DATASET_NAME_bbc_news_summary = 'bbc_news_summary'
CURRENT_DATASET = DATASET_NAME_news_category_dataset

# Logging config
CLASSIFIER_TRAIN_VERBOSE = True

# Plotting
BAR_WIDTH = 0.35

# Other
SCORE_DECIMAL_PLACES = 4
TIME_DECIMAL_PLACES = 2

Dataset = Enum('Dataset', 'ds20newsgroups '
               + 'news_category_dataset '
               + 'bbc_news_summary')
FeatureExtractionMethod = Enum('FeatureExtractionMethod',
                               'BOW TF_IDF WORD2VEC FASTTEXT')
ClassificationMethod = Enum('ClassificationMethod', 'Naive_Bayes_Classifier '
                                                    'Logistic_Regression '
                                                    'Support_Vector_Machines '
                                                    'SVM_with_SGD Random_Forest')


# def reload_constants():
#     topic_classification.experiment_config.CLASSIFIERS_AND_RESULTS_DIR_PATH = '/home/konrad/Repositories/master-diploma/' \
#                             'topic_classification/trained_classifiers/' \
#                                                                    + EXPERIMENT_NAME + '/'
#     topic_classification.experiment_config.RESULTS_PATH = topic_classification.experiment_config.CLASSIFIERS_AND_RESULTS_DIR_PATH + 'results_' + str(
#         CLASSIFIER_ITERATION) + '.pkl'
