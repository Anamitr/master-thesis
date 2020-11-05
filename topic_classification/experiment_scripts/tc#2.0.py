import warnings

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

# %matplotlib inline
import util
from topic_classification.constants import *
from topic_classification.dataset_utils import \
    load_preprocessed_bbc_news_summary, \
    get_dataset_avg_length
from topic_classification.datastructures import TrainingData
from topic_classification.display_utils import \
    get_train_test_distribution_by_labels_names, \
    create_bar_plot, create_2_bar_plot, create_cv_test_time_plots
from topic_classification.experiment_config import get_chosen_classifiers, \
    CLASSIFIERS_AND_RESULTS_DIR_PATH, RESULTS_PATH, TEST_SET_SIZE_RATIO
from topic_classification.feature_extraction_utils import \
    get_simple_bag_of_words_features
from topic_classification.train_utils import train_multiple_classifiers

warnings.filterwarnings('ignore')

# # Fetch and preprocess data or load from disk
# data_df = fetch_preprocess_and_save_bbc_news_summary()

# # Load dataset from disk
data_df = load_preprocessed_bbc_news_summary()
get_dataset_avg_length(data_df)

# # Split on train and test dataset
train_corpus, test_corpus, train_label_names, test_label_names = train_test_split(
    np.array(data_df['Clean Article']),
    np.array(data_df['Target Name']),
    TEST_SET_SIZE_RATIO, random_state=42)

cv_train_features, cv_test_features = get_simple_bag_of_words_features(train_corpus,
                                                                       test_corpus)
# # pack data in one class
training_data = TrainingData(cv_train_features, train_label_names,
                             cv_test_features, test_label_names)

# # Get classifier definitions
classifier_list, classifier_name_list, classifier_name_shortcut_list = \
    get_chosen_classifiers()

# Train and save on disk
results = train_multiple_classifiers(classifier_list, classifier_name_list,
                                     training_data)
# # Load from disk
# classifier_list = util.load_classifier_list(classifier_name_list,
#                                             CLASSIFIERS_AND_RESULTS_DIR_PATH)
# results = util.load_object(RESULTS_PATH)

# create_cv_test_time_plots(results, classifier_name_shortcut_list)
cv_mean_scores = [round(result[1], SCORE_DECIMAL_PLACES) for result in results]
test_scores = [round(result[2], SCORE_DECIMAL_PLACES) for result in results]
elapsed_times = [round(result[3], TIME_DECIMAL_PLACES) for result in results]
# create_bar_plot(classifier_name_shortcut_list, 'Classifier scores', 'Accuracy',
#                 cv_mean_scores, y_range_tuple=(0, 1))
create_2_bar_plot(classifier_name_shortcut_list, 'Classifier scores', 'Accuracy',
                  cv_mean_scores, test_scores, 'cv means', 'test set',
                  y_range_tuple=(0, 1), should_autolabel=False)
create_bar_plot(classifier_name_shortcut_list, 'Elapsed training times',
                'Time in seconds', elapsed_times, color='red')

# results_df = pd.DataFrame(
#         [[classifier_name_shortcut_list[i], results[i][1], results[i][2]] for i in
#          range(0, len(results))]).T

results_df = pd.DataFrame([[round(item[1], SCORE_DECIMAL_PLACES),
                            round(item[2], SCORE_DECIMAL_PLACES)
                            ] for item in results]).T
results_df.columns = classifier_name_shortcut_list
results_df.index = ['CV Mean', 'Test accuracy']
results_df.to_clipboard()
