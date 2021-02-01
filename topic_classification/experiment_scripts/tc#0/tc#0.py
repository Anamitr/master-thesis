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


def run_tc0_0():
    global dataset, feature_extraction_method, classifiers, experiment_controller
    dataset = Dataset.ds20newsgroups
    feature_extraction_method = FeatureExtractionMethod.BOW
    classifiers = [
        ClassificationMethod.Naive_Bayes_Classifier,
        # ClassificationMethod.Logistic_Regression,
        ClassificationMethod.Support_Vector_Machines,
        ClassificationMethod.SVM_with_SGD]

    experiment_controller = ExperimentController('tc#0.0', '1')
    experiment_controller.set_variables(dataset, feature_extraction_method,
                                        classifiers)
    experiment_controller.run_experiment()


def run_tc0_2():
    global dataset, feature_extraction_method, classifiers, experiment_controller
    dataset = Dataset.ds20newsgroups
    feature_extraction_method = FeatureExtractionMethod.WORD2VEC
    classifiers = [
        ClassificationMethod.Naive_Bayes_Classifier,
        ClassificationMethod.Logistic_Regression,
        ClassificationMethod.Support_Vector_Machines,
        ClassificationMethod.SVM_with_SGD]

    experiment_controller = ExperimentController('tc#0.2', '1')
    experiment_controller.set_variables(dataset, feature_extraction_method,
                                        classifiers,
                                        should_load_embedding_model=True)
    experiment_controller.run_experiment()
    # experiment_controller.load_results_from_disk()
    # experiment_controller.display_results()


run_tc0_0()
# run_tc0_2()
