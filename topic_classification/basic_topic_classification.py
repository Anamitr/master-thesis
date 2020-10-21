import sys

from sklearn.datasets import fetch_20newsgroups
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB

# %matplotlib inline
from topic_classification.classifiers_utils import train_multiple_classifiers
from topic_classification.dataset_utils import load_20newsgroups
from topic_classification.datastructures import TrainingData
from topic_classification.display_utils import get_train_test_distribution_by_labels_names

warnings.filterwarnings('ignore')

# Fetch and preprocess data or load from disk
# data_df = fetch_preprocess_and_save_20newsgroups()

data_df = load_20newsgroups()

# For some reason dataset still contains np.nan
data_df = data_df.dropna()

# Split on train and test dataset
train_corpus, test_corpus, train_label_nums, test_label_nums, train_label_names, \
test_label_names = train_test_split(np.array(data_df['Clean Article']),
                                    np.array(data_df['Target Label']),
                                    np.array(data_df['Target Name']),
                                    test_size=0.33, random_state=42)

# Table of category, train count, test count
train_test_distribution = get_train_test_distribution_by_labels_names(train_label_names,
                                                                      test_label_names)

# build BOW features on train articles
cv = CountVectorizer(binary=False, min_df=0.0, max_df=1.0)
cv_train_features = cv.fit_transform(train_corpus)
# transform test articles into features
cv_test_features = cv.transform(test_corpus)
print('BOW model:> Train features shape:', cv_train_features.shape,
      ' Test features shape:', cv_test_features.shape, '\n')
# pack data in one class
training_data = TrainingData(cv_train_features, train_label_names,
                             cv_test_features, test_label_names)

classifier_list = []
classifier_list.append(MultinomialNB(alpha=1))
classifier_list.append(
    LogisticRegression(penalty='l2', max_iter=100, C=1, random_state=42))
classifier_list.append(LinearSVC(penalty='l2', C=1, random_state=42))
classifier_list.append(
    SGDClassifier(loss='hinge', penalty='l2', max_iter=5, random_state=42))
classifier_list.append(RandomForestClassifier(n_estimators=10, random_state=42))
classifier_list.append(GradientBoostingClassifier(n_estimators=10, random_state=42))

classifier_name_list = []
classifier_name_list.append('Naive Bayes Classifier')
classifier_name_list.append('Logistic Regression')
classifier_name_list.append('Support Vector Machines')
classifier_name_list.append('SVM with Stochastic Gradient Descent')
classifier_name_list.append('Random Forest')
classifier_name_list.append('Gradient Boosting Machines')
results = train_multiple_classifiers(classifier_list, classifier_name_list, training_data)

