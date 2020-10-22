from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

# Statistical classifiers
multinominal_naive_bayes_classifier = \
    (MultinomialNB(alpha=1), 'Naive Bayes Classifier', 'NBC')
logistic_regression_classifier = (
    LogisticRegression(penalty='l2', max_iter=100, C=1, random_state=42),
    'Logistic Regression', 'SVM')
support_vector_machines_classifier = (LinearSVC(penalty='l2', C=1, random_state=42),
                                      'Support Vector Machines', 'SVM')
svm_with_stochastic_gradient_descent_classifier = (
    SGDClassifier(loss='hinge', penalty='l2', max_iter=5, random_state=42),
    'SVM with Stochastic Gradient Descent', 'SGD')
random_forest_classifier = (RandomForestClassifier(n_estimators=10, random_state=42),
                            'Random Forest', 'RF')
gradient_boosting_machines_classifier = (
    GradientBoostingClassifier(n_estimators=10, random_state=42),
    'Gradient Boosting Machines', 'GBC')


def get_basic_statistical_classifiers():
    classifiers_tuples = (
        multinominal_naive_bayes_classifier, logistic_regression_classifier,
        support_vector_machines_classifier,
        svm_with_stochastic_gradient_descent_classifier, random_forest_classifier,
        gradient_boosting_machines_classifier)
    classifier_list = [classifier_tuple[0] for classifier_tuple in classifiers_tuples]
    classifier_name_list = [classifier_tuple[1] for classifier_tuple in
                            classifiers_tuples]
    classifier_name_shortcut_list = [classifier_tuple[2] for classifier_tuple in
                                     classifiers_tuples]

    return classifier_list, classifier_name_list, classifier_name_shortcut_list
