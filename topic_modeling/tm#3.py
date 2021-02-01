import os
from collections import Counter
from itertools import permutations
import pyLDAvis
import pyLDAvis.sklearn

from topic_classification.constants import Dataset, ModelingMethod
from topic_classification.display_utils import create_2_bar_plot
from topic_modeling.TMExperimentController import TMExperimentController

os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'

dataset = Dataset.arxiv_metadata
NUM_OF_TOPICS = 8
modeling_method = ModelingMethod.LDA
tm_experiment_controller = TMExperimentController()


def find_best_permutation():
    # topics_in_order = ['T1', 'T2', 'econ', 'T4', 'T5', 'cs', 'physics',
    #                    'q-bio']  # LDA
    topics_in_order = ['econ', 'stat', 'T3', 'math', 'physics', 'q-bio', 'T7',
                       'T8']  # NMF
    unidentified_topics = ['eess', 'cs', 'q-fin']
    unidentified_topics_perms_list = []
    for perm in permutations(unidentified_topics):
        unidentified_topics_perms_list.append(perm)
    # topics_in_order = ['tech', 'entertainment', 'politics', 'business', 'sports']
    # NMF
    # results = {}
    for i in range(0, len(unidentified_topics_perms_list)):
        print('Perm', i, ":", unidentified_topics_perms_list[i])
        topics_in_order[2] = unidentified_topics_perms_list[i][0]
        topics_in_order[6] = unidentified_topics_perms_list[i][1]
        topics_in_order[7] = unidentified_topics_perms_list[i][2]
        # topics_in_order[7] = unidentified_topics_perms_list[i][3]
        tm_experiment_controller.set_topics_in_order(topics_in_order)
        tm_experiment_controller.test_prediction_accuracy()


def first_experiments():
    dataset = Dataset.arxiv_metadata
    NUM_OF_TOPICS = 8
    modeling_method = ModelingMethod.LDA

    tm_experiment_controller = TMExperimentController()
    tm_experiment_controller.set_variables(dataset, modeling_method, NUM_OF_TOPICS)

    # tm_experiment_controller.run_experiment()

    # # You have to manually identify topics from
    # # tm_experiment_controller.modeling_results_df

    # find_best_permutation()
    # # topics_in_order = ['q-fin', 'stat', 'econ', 'math', 'eess', 'cs', 'physics',
    # #                    'q-bio'] # LDA
    # topics_in_order = ['cs', 'stat', 'math', 'q-bio', 'eess', 'q-fin', 'econ',
    #                    'physics']  # NMF
    # tm_experiment_controller.set_topics_in_order(topics_in_order)
    #
    # # tm_experiment_controller.test_prediction_accuracy()
    # tm_experiment_controller.cal_scores_per_topic()


def second_experiment():
    global dataset, NUM_OF_TOPICS, modeling_method, tm_experiment_controller
    dataset = Dataset.arxiv_metadata
    NUM_OF_TOPICS = 7
    modeling_method = ModelingMethod.LDA

    tm_experiment_controller.set_variables(dataset, modeling_method, NUM_OF_TOPICS)
    tm_experiment_controller.run_experiment(True)

    # Test additional preprocessing
    # tm_experiment_controller.load_dataset()
    # tm_experiment_controller.do_additional_dataset_preprocessing()
    # tm_experiment_controller.get_dataset_categories()

    #
    # # LDA
    # topics_in_order = ['cs', 'q-bio', 'econ-q-fin', 'math-stat', 'eess-physics']
    # topics_in_order = ['stat', 'math']
    # topics_in_order = ['econ', 'stat', 'q-fin', 'math', 'physics', 'q-bio', 'eess',
    #                    'cs']
    topics_in_order = ['econ-q-fin', 'stat', 'q-bio', 'math', 'physics', 'eess',
                       'cs']
    tm_experiment_controller.set_topics_in_order(topics_in_order)
    tm_experiment_controller.test_prediction_accuracy()
    tm_experiment_controller.cal_scores_per_topic()


def plot_lda():
    pyLDAvis.enable_notebook()
    vis_data = pyLDAvis.sklearn.prepare(tm_experiment_controller.lda_model,
                                        tm_experiment_controller.train_features,
                                        tm_experiment_controller.cv, mds='mmds')
    pyLDAvis.show(vis_data)


def plot_comparison():
    p1 = ['BBC News Summary', 'ArXiv Dataset']
    p2 = ''
    p3 = 'Celność testowa'
    p4 = [0.9809, 0.8542]
    p5 = [0.9182, 0.619]
    p6 = 'uczenie nadzorowane'
    p7 = 'uczenie nienadzorowane'
    create_2_bar_plot(p1, p2, p3, p4, p5, p6, p7)


second_experiment()
