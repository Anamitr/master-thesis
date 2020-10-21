import pickle

from topic_classification.constants import CLASSIFIER_ITERATION


def save_string_to_file(output_file_name: str, content: str):
    output_file = open(output_file_name, "w")
    output_file.write(content)
    output_file.close()


def save_classifier_list(classifier_list, filename_list, path):
    for i in range(0, len(classifier_list)):
        save_classifier(classifier_list[i], path + filename_list[i].replace(' ', '_')
                        + '_' + CLASSIFIER_ITERATION + '.pkl')


def save_classifier(classifier, filename_with_path):
    pickle.dump(classifier, open(filename_with_path, 'wb'))


def load_classifier(filename_with_path):
    return pickle.load(open(filename_with_path, 'rb'))
