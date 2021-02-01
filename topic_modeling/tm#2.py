import sys

from topic_classification.constants import Dataset, ModelingMethod, \
    FeatureExtractionMethod
from topic_modeling.TMExperimentController import TMExperimentController

dataset = Dataset.bbc_news_summary
NUM_OF_TOPICS = 5
modeling_method = ModelingMethod.NMF

tm_experiment_controller = TMExperimentController()
tm_experiment_controller.set_variables(dataset,
                                       modeling_method, NUM_OF_TOPICS)
tm_experiment_controller.run_experiment()

# # You have to manually identify topics from
# # tm_experiment_controller.modeling_results_df
topics_in_order = ['sport', 'tech', 'politics', 'business', 'entertainment']  # LDA
topics_in_order = ['tech', 'entertainment', 'politics', 'business', 'sports']  # NMF
tm_experiment_controller.set_topics_in_order(topics_in_order)
tm_experiment_controller.test_prediction_accuracy()
