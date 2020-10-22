from sklearn.model_selection import cross_val_score
import numpy as np
import time

from topic_classification.datastructures import TrainingData


def train_multiple_classifiers(classifier_list, classifier_name_list, training_data:
TrainingData):
    if len(classifier_list) != len(classifier_name_list):
        print("Classifier list length and classifier name list length must be equal!")
        return

    results_list = []
    for i in range(0, len(classifier_list)):
        results_list.append(train_classifier_and_display_results(classifier_list[i],
                                                            classifier_name_list[i],
                                                            training_data))
    return results_list


def train_classifier_and_display_results(classifier, classifier_name: str, training_data:
TrainingData):
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
                                        training_data.train_label_names,
                                        cv=5)
    mnb_bow_cv_mean_score = np.mean(mnb_bow_cv_scores)

    mnb_bow_test_score = classifier.score(training_data.cv_test_features,
                                          training_data.test_label_names)

    return [mnb_bow_cv_scores, mnb_bow_cv_mean_score, mnb_bow_test_score]
