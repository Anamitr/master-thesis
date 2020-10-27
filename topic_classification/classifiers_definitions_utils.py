from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from topic_classification.constants import *

# Statistical classifiers

multinominal_naive_bayes_classifier = \
    (MultinomialNB(alpha=1), 'Naive Bayes Classifier', 'NBC')
logistic_regression_classifier = (
    LogisticRegression(penalty='l2', max_iter=100, C=1, random_state=42,
                       verbose=CLASSIFIER_TRAIN_VERBOSE),
    'Logistic Regression', 'LR')
support_vector_machines_classifier = (
    LinearSVC(penalty='l2', C=1, random_state=42, verbose=CLASSIFIER_TRAIN_VERBOSE),
    'Support Vector Machines', 'SVM')
svm_with_stochastic_gradient_descent_classifier = (
    SGDClassifier(loss='hinge', penalty='l2', max_iter=5, random_state=42,
                  verbose=CLASSIFIER_TRAIN_VERBOSE),
    'SVM with Stochastic Gradient Descent', 'SGD')
random_forest_classifier = (RandomForestClassifier(n_estimators=10,
                                                   random_state=42,
                                                   verbose=CLASSIFIER_TRAIN_VERBOSE),
                            'Random Forest', 'RF')
gradient_boosting_machines_classifier = (
    GradientBoostingClassifier(n_estimators=10, random_state=42,
                               verbose=CLASSIFIER_TRAIN_VERBOSE),
    'Gradient Boosting Machines', 'GBC',)


