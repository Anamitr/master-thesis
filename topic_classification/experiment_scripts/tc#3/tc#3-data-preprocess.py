import json
import pandas as pd

from topic_classification.ExperimentController import ExperimentController
from topic_classification.constants import Dataset
from topic_classification.dataset_utils import \
    fetch_and_preprocess_arxiv_metadata_dataset

experiment_controller = ExperimentController()

dataset = Dataset.arxiv_metadata

data_df = fetch_and_preprocess_arxiv_metadata_dataset()

# documents = []
# for line in open(experiment_controller.TOPIC_CLASSIFICATION_DATA_PATH +
#                  dataset.name + '.json', 'r'):
#     documents.append(json.loads(line))
#
#
# def strip_category_from_subcategories(category_string: str):
#     return category_string.split(' ')[0].split('.')[0]
#
#
# data = {'Article': [document['abstract'] for document in documents],
#         'Target Name': [strip_category_from_subcategories(document['categories'])
#                         for document in documents]}
#
# data_df = pd.DataFrame(data, columns=['Article', 'Target Name'])
#
# num_of_categories = len(set(data['Target Name']))
# print('Dataset ', dataset.name, ' num of categories:', num_of_categories)
