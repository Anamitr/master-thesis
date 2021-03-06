import fasttext
import numpy as np
from sklearn.model_selection import train_test_split

# %matplotlib inline
import text_preprocessing.text_normalizer as tn
import topic_classification.experiment_config as experiment_config
from topic_classification.constants import *
from topic_classification.dataset_utils import \
    load_preprocessed_news_category_dataset
from topic_classification.datastructures import TrainingData
from topic_classification.display_utils import \
    create_bar_plot, create_2_bar_plot
from topic_classification.feature_extraction_utils import \
    document_vectorize_with_fasttext_model
from topic_classification.train_utils import train_multiple_classifiers, \
    get_chosen_classifiers

data_df = load_preprocessed_news_category_dataset()

train_corpus, test_corpus, train_label_names, \
test_label_names = train_test_split(np.array(data_df['Clean Article']),
                                    np.array(data_df['Target Name']),
                                    test_size=0.33, random_state=42)
# tokenize corpus
tokenized_train = [tn.tokenizer.tokenize(text) for text in train_corpus]
tokenized_test = [tn.tokenizer.tokenize(text) for text in test_corpus]

# Get list of words (I know, cool, not readable one-liner)
data_word_list = ''.join(list(data_df['Clean Article'])).split(' ')
vocabulary = set(data_word_list)

ft_num_features = 1000
# # Train and save FastText model

# fasttext_model = fasttext.train_unsupervised(TOPIC_CLASSIFICATION_DATA_PATH +
#                                              '20_newsgroups_one_string.txt',
#                                              epoch=100,
#                                              dim=ft_num_features)
# fasttext_model.save_model(experiment_config.FAST_TEXT_SAVE_PATH)

# Load FastText model
fasttext_model = fasttext.load_model(experiment_config.FAST_TEXT_SAVE_PATH)

avg_ft_train_features = document_vectorize_with_fasttext_model(
    corpus=tokenized_train,
    fasttext_model=fasttext_model,
    num_features=ft_num_features,
    vocabulary=vocabulary)
avg_ft_test_features = document_vectorize_with_fasttext_model(
    corpus=tokenized_test,
    fasttext_model=fasttext_model,
    num_features=ft_num_features,
    vocabulary=vocabulary)

training_data = TrainingData(avg_ft_train_features, train_label_names,
                             avg_ft_test_features, test_label_names)

classifier_list, classifier_name_list, classifier_name_shortcut_list = \
    get_chosen_classifiers()


def train_and_save(classifier_list, classifier_name_list, training_data):
    results = train_multiple_classifiers(classifier_list, classifier_name_list,
                                         training_data)
    # util.save_object(results, RESULTS_PATH)
    # util.save_classifier_list(classifier_list, classifier_name_list,
    #                           CLASSIFIERS_AND_RESULTS_DIR_PATH)
    return results


results = train_and_save(classifier_list, classifier_name_list, training_data)

mean_scores = [round(result[1], SCORE_DECIMAL_PLACES) for result in results]
test_scores = [round(result[2], SCORE_DECIMAL_PLACES) for result in results]
elapsed_times = [round(result[3], TIME_DECIMAL_PLACES) for result in results]

create_2_bar_plot(classifier_name_shortcut_list, 'Classifier scores', 'Accuracy',
                  mean_scores, test_scores, 'cv means', 'test set',
                  y_range_tuple=(0, 1))
create_bar_plot(classifier_name_shortcut_list, 'Elapsed training times',
                'Time in seconds', elapsed_times, color='red')

# LR
# CV Accuracy (5-fold): [0.53598633 0.53020957 0.53526308 0.53578329 0.53255054]
# Mean CV Accuracy: 0.5339585613382408 Test Accuracy: 0.5334731505650527
# Time elapsed: 1860.7 seconds
