# Experiment naming
from topic_classification.classifiers_definitions_utils import \
    multinominal_naive_bayes_classifier, logistic_regression_classifier, \
    support_vector_machines_classifier, \
    svm_with_stochastic_gradient_descent_classifier, random_forest_classifier, \
    gradient_boosting_machines_classifier

EXPERIMENT_NAME = 'tc#0.3'
CLASSIFIER_ITERATION = 2
classifiers_tuples = (
    # multinominal_naive_bayes_classifier,
    logistic_regression_classifier,
    # support_vector_machines_classifier,
    # svm_with_stochastic_gradient_descent_classifier,
    # random_forest_classifier,
    # gradient_boosting_machines_classifier
)


def get_basic_statistical_classifiers():
    classifier_list = [classifier_tuple[0] for classifier_tuple in
                       classifiers_tuples]
    classifier_name_list = [classifier_tuple[1] for classifier_tuple in
                            classifiers_tuples]
    classifier_name_shortcut_list = [classifier_tuple[2] for classifier_tuple in
                                     classifiers_tuples]

    return classifier_list, classifier_name_list, classifier_name_shortcut_list


# Save paths
CLASSIFIERS_SAVE_PATH = '/home/konrad/Repositories/master-diploma/' \
                        'topic_classification/trained_classifiers/' \
                        + EXPERIMENT_NAME + '/'
RESULTS_PATH = CLASSIFIERS_SAVE_PATH + 'results_' + str(
    CLASSIFIER_ITERATION) + '.pkl'
WORD2VEC_MODEL_SAVE_PATH = CLASSIFIERS_SAVE_PATH + 'w2v_model_' + str(
    CLASSIFIER_ITERATION) + '.pkl'
