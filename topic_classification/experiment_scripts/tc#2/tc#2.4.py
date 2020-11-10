import warnings

# %matplotlib inline
from topic_classification.ExperimentController import ExperimentController
from topic_classification.constants import *

warnings.filterwarnings('ignore')
# Script for different kind of experiments

dataset = Dataset.bbc_news_summary
feature_extraction_method = FeatureExtractionMethod.FASTTEXT
classifiers = [
    ClassificationMethod.Logistic_Regression,
    ClassificationMethod.Support_Vector_Machines,
    ClassificationMethod.SVM_with_SGD]

experiment_controller = ExperimentController('tc#2.4', '1')
experiment_controller.run_experiment(dataset, feature_extraction_method,
                                     classifiers, should_load_embedding_model=True)
