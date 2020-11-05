import warnings

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

# %matplotlib inline
import util
import topic_classification.constants as constants
from topic_classification.experiment_config import get_chosen_classifiers, \
    CLASSIFIERS_AND_RESULTS_DIR_PATH, RESULTS_PATH
from topic_classification.constants import *
from topic_classification.dataset_utils import load_20newsgroups
from topic_classification.datastructures import TrainingData
from topic_classification.display_utils import \
    get_train_test_distribution_by_labels_names, \
    create_bar_plot, create_2_bar_plot
from topic_classification.train_utils import train_multiple_classifiers

warnings.filterwarnings('ignore')

topic_classification.experiment_config.EXPERIMENT_NAME = 'tc#0'
constants.reload_constants()
from topic_classification.constants import *

# # Fetch and preprocess data or load from disk
# data_df = fetch_preprocess_and_save_20newsgroups()

# # Load dataset from disk
data_df = load_20newsgroups()

# # For some reason dataset still contains np.nan
data_df = data_df.dropna()

# # Split on train and test dataset
train_corpus, test_corpus, train_label_names, test_label_names = train_test_split(
    np.array(data_df['Clean Article']),
    np.array(data_df['Target Name']),
    test_size=0.33, random_state=42)

# # Display table of category, train count, test count
train_test_distribution = get_train_test_distribution_by_labels_names(
    train_label_names, test_label_names)

# # build BOW features on train articles
cv = CountVectorizer(binary=False, min_df=0.0, max_df=1.0)
cv_train_features = cv.fit_transform(train_corpus)
# # transform test articles into features
cv_test_features = cv.transform(test_corpus)
print('BOW model:> Train features shape:', cv_train_features.shape,
      ' Test features shape:', cv_test_features.shape, '\n')
# # pack data in one class
training_data = TrainingData(cv_train_features, train_label_names,
                             cv_test_features, test_label_names)

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


# Train and save on disk
# results = train_and_save(classifier_list, classifier_name_list, training_data)
# #Load from disk
classifier_list = util.load_classifier_list(classifier_name_list,
                                            CLASSIFIERS_AND_RESULTS_DIR_PATH)
results = util.load_object(RESULTS_PATH)

# results[0] = array of crossvalidation, [1] crossvalidation scores,
# [2] test score, [3] times
# # Plotting
cv_mean_scores = [round(result[1], SCORE_DECIMAL_PLACES) for result in results]
test_scores = [round(result[2], SCORE_DECIMAL_PLACES) for result in results]
elapsed_times = [round(result[3], TIME_DECIMAL_PLACES) for result in results]
# create_bar_plot(classifier_name_shortcut_list, 'Classifier scores', 'Accuracy',
#                 cv_mean_scores, y_range_tuple=(0, 1))
create_2_bar_plot(classifier_name_shortcut_list, 'Classifier scores', 'Accuracy',
                  cv_mean_scores, test_scores, 'cv means', 'test set',
                  y_range_tuple=(0, 1))
create_bar_plot(classifier_name_shortcut_list, 'Elapsed training times',
                'Time in seconds', elapsed_times, color='red')
