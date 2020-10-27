import errno
import os
import pickle

from topic_classification.experiment_config import CLASSIFIER_ITERATION


# Persistence
def save_string_to_file(output_file_name: str, content: str):
    output_file = open(output_file_name, "w")
    output_file.write(content)
    output_file.close()


def save_classifier_list(classifier_list, filename_list, path):
    create_path_if_not_exists(path)
    for i in range(0, len(classifier_list)):
        save_object(classifier_list[i],
                    path + convert_name_to_filename(filename_list[i])
                    + '_' + str(CLASSIFIER_ITERATION) + '.pkl')


def load_classifier_list(filename_list, path):
    classifier_list = []
    for i in range(0, len(filename_list)):
        classifier_list.append(
            load_object(path + convert_name_to_filename(filename_list[i])
                        + '_' + str(CLASSIFIER_ITERATION) + '.pkl'))


def save_object(object_to_save, filename_with_path):
    create_path_if_not_exists(filename_with_path)
    pickle.dump(object_to_save, open(filename_with_path, 'wb'))


def load_object(filename_with_path):
    return pickle.load(open(filename_with_path, 'rb'))


def create_path_if_not_exists(path):
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


def load_results_of_specific_topic_classification_experiment(experiment_name: str,
                                                             classifier_iteration):
    results_save_dir_path = '/home/konrad/Repositories/master-diploma/' \
                            'topic_classification/trained_classifiers/' \
                            + experiment_name + '/'
    results_path = results_save_dir_path + 'results_' + str(
        classifier_iteration) + '.pkl'
    return load_object(results_path)


# Other
def convert_name_to_filename(name: str):
    return name.replace(' ', '_').lower()


def transpose_list(list_to_transpose: list):
    return list(map(list, zip(*list_to_transpose)))
