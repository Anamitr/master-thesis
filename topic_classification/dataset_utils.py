import json
import os

import pandas as pd
from sklearn.datasets import fetch_20newsgroups

import text_preprocessing.text_normalizer as tn
from topic_classification.constants import *


def fetch_preprocess_and_save_20newsgroups():
    data = fetch_20newsgroups(subset='all', shuffle=True,
                              remove=('headers', 'footers', 'quotes'))
    data_labels_map = dict(enumerate(data.target_names))

    corpus, target_labels, target_names = (data.data, data.target,
                                           [data_labels_map[label] for label in
                                            data.target])
    data_df = pd.DataFrame(
        {'Article': corpus, 'Target Label': target_labels,
         'Target Name': target_names})
    print(data_df.shape)

    # Preprocessing
    total_nulls = data_df[data_df.Article.str.strip() == ''].shape[0]
    print("Empty documents:", total_nulls)

    data_df = data_df[~(data_df.Article.str.strip() == '')]
    print("data_df.shape: ", data_df.shape)

    import nltk
    stopword_list = nltk.corpus.stopwords.words('english')
    # just to keep negation if any in bi-grams
    stopword_list.remove('no')
    stopword_list.remove('not')

    # normalize our corpus
    norm_corpus = tn.normalize_corpus(corpus=data_df['Article'], html_stripping=True,
                                      contraction_expansion=True,
                                      accented_char_removal=True,
                                      text_lower_case=True,
                                      text_lemmatization=True,
                                      text_stemming=False, special_char_removal=True,
                                      remove_digits=True,
                                      stopword_removal=True, stopwords=stopword_list)
    data_df['Clean Article'] = norm_corpus
    data_df.to_csv(
        TOPIC_CLASSIFICATION_DATA_PATH + DATASET_NAME_20newsgroups
        + '_preprocessed.csv',
        index=False)
    return data_df


def load_20newsgroups():
    data_df = pd.read_csv(
        TOPIC_CLASSIFICATION_DATA_PATH + DATASET_NAME_20newsgroups
        + '_preprocessed.csv')
    # # For some reason dataset still contains np.nan
    data_df = data_df.dropna()
    return data_df


def fetch_preprocess_and_save_news_category_dataset():
    df = pd.read_json(
        TOPIC_CLASSIFICATION_DATA_PATH + 'News_Category_Dataset_v2.json',
        lines=True)
    # Merge two same categories
    df.category = df.category.map(
        lambda x: "WORLDPOST" if x == "THE WORLDPOST" else x)
    df['text'] = df.headline + " " + df.short_description
    data_df = pd.DataFrame(
        {'Article': df['text'], 'Target Name': df['category']})
    data_df = preprocess_data_frame(data_df)
    data_df.to_csv(
        TOPIC_CLASSIFICATION_DATA_PATH + CURRENT_DATASET + '_preprocessed.csv',
        index=False)
    return data_df


def load_preprocessed_news_category_dataset():
    return pd.read_csv(
        TOPIC_CLASSIFICATION_DATA_PATH + DATASET_NAME_news_category_dataset +
        '_preprocessed.csv').dropna()


def fetch_preprocess_and_save_bbc_news_summary():
    data_path = TOPIC_CLASSIFICATION_DATA_PATH + 'BBC News Summary/News Articles/'
    print("Collecting data.... ")
    data = []
    count = 0

    for dir in os.listdir(data_path):
        for file in os.listdir(data_path + dir):
            try:
                text = ''
                name = file
                myfile = open(data_path + dir + '/' + file, "r")
                text = myfile.read()
                count += 1
                data.append([text, dir])
            except:
                continue
        print(str(count) + " text files found in " + dir + " folder.")
        count = 0

    print("Data loaded")
    data_df = pd.DataFrame(data, columns=['Article', 'Target Name'])
    data_df = preprocess_data_frame(data_df)
    data_df.to_csv(
        TOPIC_CLASSIFICATION_DATA_PATH + CURRENT_DATASET + '_preprocessed.csv',
        index=False)
    return data_df


def load_preprocessed_bbc_news_summary():
    return pd.read_csv(
        TOPIC_CLASSIFICATION_DATA_PATH + DATASET_NAME_bbc_news_summary +
        '_preprocessed.csv').dropna()


def fetch_and_preprocess_arxiv_metadata_dataset():
    dataset = Dataset.arxiv_metadata
    print('Fetching and preprocessing dataset', dataset.name)

    documents = []
    for line in open(TOPIC_CLASSIFICATION_DATA_PATH +
                     dataset.name + '.json', 'r'):
        documents.append(json.loads(line))

    def strip_category_from_subcategories(category_string: str):
        return category_string.split(' ')[0].split('.')[0]

    data = {'Article': [document['abstract'] for document in documents],
            'Target Name': [strip_category_from_subcategories(document['categories'])
                            for document in documents]}

    data_df = pd.DataFrame(data, columns=['Article', 'Target Name'])

    num_of_categories = len(set(data['Target Name']))
    print('Num of categories:', num_of_categories)

    data_df = preprocess_data_frame(data_df)
    data_df.to_csv(
        TOPIC_CLASSIFICATION_DATA_PATH + dataset.name + '_preprocessed.csv',
        index=False)
    return data_df


def load_preprocessed_arxiv_metadata_dataset():
    dataset_dir_path = TOPIC_CLASSIFICATION_DATA_PATH + DATASET_NAME_arxiv_metadata \
                       + '_preprocessed/'
    data_list = []
    for filename in os.listdir(dataset_dir_path):
        if filename.endswith('.csv'):
            data_list.append(pd.read_csv(dataset_dir_path + filename))
    return pd.concat(data_list, axis=0, ignore_index=True)


def preprocess_data_frame(data_df: pd.DataFrame):
    # Preprocessing
    total_nulls = data_df[data_df.Article.str.strip() == ''].shape[0]
    print("Empty documents:", total_nulls)

    data_df = data_df[~(data_df.Article.str.strip() == '')]
    print("data_df.shape: ", data_df.shape)

    import nltk
    stopword_list = nltk.corpus.stopwords.words('english')
    # just to keep negation if any in bi-grams
    stopword_list.remove('no')
    stopword_list.remove('not')

    # normalize our corpus
    norm_corpus = tn.normalize_corpus(corpus=data_df['Article'], html_stripping=True,
                                      contraction_expansion=True,
                                      accented_char_removal=True,
                                      text_lower_case=True,
                                      text_lemmatization=True,
                                      text_stemming=False, special_char_removal=True,
                                      remove_digits=True,
                                      stopword_removal=True, stopwords=stopword_list)
    data_df['Clean Article'] = norm_corpus
    return data_df


def get_dataset_avg_length(data_df: pd.DataFrame):
    text_lengths = [len(text) for text in data_df['Clean Article']]
    average_text_length = sum(text_lengths) / len(text_lengths)
    print('Average text length:', round(average_text_length))
    return average_text_length
