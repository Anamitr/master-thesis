from sklearn.model_selection import cross_val_score
import numpy as np
import time

import util
from topic_classification.datastructures import TrainingData


# from topic_classification.experiment_config import CLASSIFIERS_AND_RESULTS_DIR_PATH, \
#     CLASSIFIER_ITERATION, RESULTS_PATH


def train_multiple_classifiers(classifier_list, classifier_name_list, training_data:
TrainingData, CLASSIFIERS_AND_RESULTS_DIR_PATH, CLASSIFIER_ITERATION, RESULTS_PATH):
    if len(classifier_list) != len(classifier_name_list):
        print(
            "Classifier list length and classifier name list length must be equal!")
        return

    results_list = []
    for i in range(0, len(classifier_list)):
        results = train_classifier_and_display_results(classifier_list[i],
                                                       classifier_name_list[i],
                                                       training_data)
        results_list.append(results)

        util.save_object(results,
                         CLASSIFIERS_AND_RESULTS_DIR_PATH +
                         util.convert_name_to_filename(classifier_name_list[i])
                         + '_' + str(CLASSIFIER_ITERATION) + '_results.pkl')
        util.save_object(classifier_list[i],
                         CLASSIFIERS_AND_RESULTS_DIR_PATH +
                         util.convert_name_to_filename(classifier_name_list[i])
                         + '_' + str(CLASSIFIER_ITERATION) + '.pkl')
    util.save_object(results_list, RESULTS_PATH)
    return results_list


def train_classifier_and_display_results(classifier, classifier_name: str,
                                         training_data: TrainingData):
    print(classifier_name)
    start_time = time.time()
    results = train_classifier_with_count_vectorizer(classifier, training_data)
    end_time = time.time()
    print('CV Accuracy (5-fold):', results[0])
    print('Mean CV Accuracy:', results[1], 'Test Accuracy:', results[2])
    time_elapsed = round(end_time - start_time, 2)
    print('Time elapsed:', time_elapsed, 'seconds\n')
    results.append(time_elapsed)
    return results


def train_classifier_with_count_vectorizer(classifier, training_data: TrainingData):
    classifier.fit(training_data.cv_train_features, training_data.train_label_names)
    mnb_bow_cv_scores = cross_val_score(classifier, training_data.cv_train_features,
                                        training_data.train_label_names, cv=5)
    mnb_bow_cv_mean_score = np.mean(mnb_bow_cv_scores)

    mnb_bow_test_score = classifier.score(training_data.cv_test_features,
                                          training_data.test_label_names)

    return [mnb_bow_cv_scores, mnb_bow_cv_mean_score, mnb_bow_test_score]


def get_chosen_classifiers(classifiers_tuples):
    classifier_list = [classifier_tuple[0] for classifier_tuple in
                       classifiers_tuples]
    classifier_name_list = [classifier_tuple[1] for classifier_tuple in
                            classifiers_tuples]
    classifier_name_shortcut_list = [classifier_tuple[2] for classifier_tuple in
                                     classifiers_tuples]

    return classifier_list, classifier_name_list, classifier_name_shortcut_list
