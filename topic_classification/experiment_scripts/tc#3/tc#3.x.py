import warnings

# %matplotlib inline
from topic_classification.ExperimentController import ExperimentController
from topic_classification.constants import *
from topic_classification.dataset_utils import \
    load_preprocessed_arxiv_metadata_dataset

warnings.filterwarnings('ignore')

# Script for different kind of experiments
dataset = None
feature_extraction_method = None
classifiers = None
experiment_controller = None


def run_tc3_0():
    global dataset, feature_extraction_method, classifiers, experiment_controller
    dataset = Dataset.arxiv_metadata
    feature_extraction_method = FeatureExtractionMethod.BOW
    classifiers = [
        ClassificationMethod.Naive_Bayes_Classifier,
        ClassificationMethod.Logistic_Regression,
        ClassificationMethod.Support_Vector_Machines,
        ClassificationMethod.SVM_with_SGD,
        ClassificationMethod.Gradient_Boosting_Machines]

    experiment_controller = ExperimentController('tc#3.0', '1')
    experiment_controller.run_experiment(dataset, feature_extraction_method,
                                         classifiers,
                                         should_load_embedding_model=False)


def run_tc3_1():
    global dataset, feature_extraction_method, classifiers, experiment_controller
    dataset = Dataset.arxiv_metadata
    feature_extraction_method = FeatureExtractionMethod.TF_IDF
    classifiers = [
        ClassificationMethod.Naive_Bayes_Classifier,
        ClassificationMethod.Logistic_Regression,
        ClassificationMethod.Support_Vector_Machines,
        ClassificationMethod.SVM_with_SGD]

    experiment_controller = ExperimentController('tc#3.1', '1')
    experiment_controller.run_experiment(dataset, feature_extraction_method,
                                         classifiers,
                                         should_load_embedding_model=False)


def run_tc3_2():
    global dataset, feature_extraction_method, classifiers, experiment_controller
    dataset = Dataset.arxiv_metadata
    feature_extraction_method = FeatureExtractionMethod.WORD2VEC
    classifiers = [
        ClassificationMethod.Naive_Bayes_Classifier,
        ClassificationMethod.Logistic_Regression,
        ClassificationMethod.Support_Vector_Machines,
        ClassificationMethod.SVM_with_SGD]

    experiment_controller = ExperimentController('tc#3.2', '1')
    experiment_controller.run_experiment(dataset, feature_extraction_method,
                                         classifiers,
                                         should_load_embedding_model=False)


run_tc3_2()
# print('Hi')

# from topic_classification.display_utils import create_2_bar_plot, \
#     create_bar_plot
# create_2_bar_plot(experiment_controller.classifier_name_shortcut_list, 'Classifier scores',
#                   'Accuracy',
#                   experiment_controller.cv_mean_scores, experiment_controller.test_scores, 'cv means', 'test set',
#                   y_range_tuple=(0, 1), should_autolabel=True)
# create_bar_plot(experiment_controller.classifier_name_shortcut_list, 'Elapsed training times',
# 'Time in seconds', experiment_controller.elapsed_times, color='red')
