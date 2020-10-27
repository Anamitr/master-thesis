# Datasets

TOPIC_CLASSIFICATION_DATA_PATH = '/home/konrad/Repositories/master-diploma/' \
                                 'topic_classification/topic_class_data/'
DATASET_NAME_20newsgroups = '20newsgroups'
DATASET_NAME_news_category_dataset = 'news_category_dataset'
CURRENT_DATASET = DATASET_NAME_news_category_dataset

# Logging config
CLASSIFIER_TRAIN_VERBOSE = True

# Plotting
BAR_WIDTH = 0.35

# Other
SCORE_DECIMAL_PLACES = 4
TIME_DECIMAL_PLACES = 2


# def reload_constants():
#     topic_classification.experiment_config.CLASSIFIERS_SAVE_PATH = '/home/konrad/Repositories/master-diploma/' \
#                             'topic_classification/trained_classifiers/' \
#                                                                    + EXPERIMENT_NAME + '/'
#     topic_classification.experiment_config.RESULTS_PATH = topic_classification.experiment_config.CLASSIFIERS_SAVE_PATH + 'results_' + str(
#         CLASSIFIER_ITERATION) + '.pkl'
