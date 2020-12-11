import os
from collections import Counter
from itertools import permutations

from topic_classification.constants import Dataset, ModelingMethod
from topic_modeling.TMExperimentController import TMExperimentController

os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'

dataset = Dataset.arxiv_metadata
NUM_OF_TOPICS = 8
modeling_method = ModelingMethod.LDA

tm_experiment_controller = TMExperimentController()
tm_experiment_controller.set_variables(dataset, modeling_method, NUM_OF_TOPICS)
tm_experiment_controller.run_experiment()


# # You have to manually identify topics from
# # tm_experiment_controller.modeling_results_df
def find_best_permutation():
    topics_in_order = ['T1', 'T2', 'econ', 'T4', 'T5', 'cs', 'physics',
                       'q-bio']  # LDA
    unidentified_topics = ['eess', 'math', 'q-fin', 'stat']
    unidentified_topics_perms_list = []
    for perm in permutations(unidentified_topics):
        unidentified_topics_perms_list.append(perm)
    # topics_in_order = ['tech', 'entertainment', 'politics', 'business', 'sports']  # NMF
    # results = {}
    for i in range(0, len(unidentified_topics_perms_list)):
        print('Perm', i)
        topics_in_order[0] = unidentified_topics_perms_list[i][0]
        topics_in_order[1] = unidentified_topics_perms_list[i][1]
        topics_in_order[3] = unidentified_topics_perms_list[i][2]
        topics_in_order[4] = unidentified_topics_perms_list[i][3]
        tm_experiment_controller.set_topics_in_order(topics_in_order)
        tm_experiment_controller.test_prediction_accuracy()
    # Accuracy = 0.4623065617705542


topics_in_order = ['q-fin', 'stat', 'econ', 'math', 'eess', 'cs', 'physics',
                   'q-bio']
