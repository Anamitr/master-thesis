import pickle

from topic_classification.constants import CLASSIFIER_ITERATION


# Persistence
def save_string_to_file(output_file_name: str, content: str):
    output_file = open(output_file_name, "w")
    output_file.write(content)
    output_file.close()


def save_classifier_list(classifier_list, filename_list, path):
    for i in range(0, len(classifier_list)):
        save_object(classifier_list[i], path + filename_list[i].replace(' ', '_')
                    + '_' + CLASSIFIER_ITERATION + '.pkl')


def load_classifier_list(filename_list, path):
    classifier_list = []
    for i in range(0, len(filename_list)):
        classifier_list.append(load_object(path + filename_list[i].replace(' ', '_')
                                           + '_' + str(CLASSIFIER_ITERATION) + '.pkl'))


def save_object(object_to_save, filename_with_path):
    pickle.dump(object_to_save, open(filename_with_path, 'wb'))


def load_object(filename_with_path):
    return pickle.load(open(filename_with_path, 'rb'))

# Other
# def replace_spaces_with_underscores_in_string_list(string_list : list):
#     return [text]
