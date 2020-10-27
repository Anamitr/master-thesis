from sklearn.datasets import fetch_20newsgroups
import pandas as pd

import text_preprocessing.text_normalizer as tn

from topic_classification.constants import TOPIC_CLASSIFICATION_DATA_PATH, \
    DATASET_NAME_20newsgroups
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
