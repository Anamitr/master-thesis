from topic_classification.constants import Dataset, ModelingMethod
from topic_modeling.TMExperimentController import TMExperimentController

dataset = Dataset.arxiv_metadata
NUM_OF_TOPICS = 8
modeling_method = ModelingMethod.LDA

tm_experiment_controller = TMExperimentController()
tm_experiment_controller.set_variables(dataset, modeling_method, NUM_OF_TOPICS)
tm_experiment_controller.run_experiment()
