import warnings

# %matplotlib inline
from topic_classification.ExperimentController import ExperimentController
from topic_classification.constants import *
from topic_classification.dataset_utils import \
    load_preprocessed_arxiv_metadata_dataset

warnings.filterwarnings('ignore')
# Script for different kind of experiments

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
                                     classifiers, should_load_embedding_model=False)

# data_df = load_preprocessed_arxiv_metadata_dataset()
