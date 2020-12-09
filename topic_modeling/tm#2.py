from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer
import os
import gensim
import nltk
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from topic_classification.ExperimentController import ExperimentController
from topic_classification.constants import Dataset, FeatureExtractionMethod
from topic_classification.dataset_utils import load_preprocessed_bbc_news_summary

# dataset = Dataset.bbc_news_summary
# feature_extraction_method = FeatureExtractionMethod.FASTTEXT
experiment_controller = ExperimentController('tm#2', '1')

TOTAL_TOPICS = 5
top_terms = 20

topics = ['business', 'entertainment', 'politics', 'sport', 'tech']
topic_by_order = ['sport', 'tech', 'politics', 'business', 'entertainment']

lsi_model = None
lda_model = None
nmf_model = None
document_topics = None

# # # Load preprocessed dataset
data_df = load_preprocessed_bbc_news_summary(
    experiment_controller.TOPIC_CLASSIFICATION_DATA_PATH)

train_corpus, test_corpus, train_label_names, \
test_label_names = train_test_split(np.array(data_df['Clean Article']),
                                    np.array(data_df['Target Name']),
                                    test_size=0.33, random_state=42)

train_articles = [art_str.split(' ') for art_str in train_corpus]
test_articles = [art_str.split(' ') for art_str in test_corpus]

# # # Feature Extraction
cv = CountVectorizer(min_df=20, max_df=0.6, ngram_range=(1, 2),
                     token_pattern=None, tokenizer=lambda doc: doc,
                     preprocessor=lambda doc: doc)
cv_train_features = cv.fit_transform(train_articles)
cv_test_features = cv.transform(test_articles)
vocabulary = np.array(cv.get_feature_names())


def run_lsi():
    global lsi_model, document_topics
    # # # Modeling
    lsi_model = TruncatedSVD(n_components=TOTAL_TOPICS, n_iter=500, random_state=42)
    document_topics = lsi_model.fit_transform(cv_train_features)


def display_lsi_results():
    # # # Display results
    topic_terms = lsi_model.components_
    topic_terms.shape

    topic_key_term_idxs = np.argsort(-np.absolute(topic_terms), axis=1)[:,
                          :top_terms]
    topic_keyterm_weights = np.array([topic_terms[row, columns]
                                      for row, columns in list(
            zip(np.arange(TOTAL_TOPICS), topic_key_term_idxs))])
    topic_keyterms = vocabulary[topic_key_term_idxs]
    topic_keyterms_weights = list(zip(topic_keyterms, topic_keyterm_weights))
    for n in range(TOTAL_TOPICS):
        print('Topic #' + str(n + 1) + ':')
        print('=' * 50)
        d1 = []
        d2 = []
        terms, weights = topic_keyterms_weights[n]
        term_weights = sorted([(t, w) for t, w in zip(terms, weights)],
                              key=lambda row: -abs(row[1]))
        for term, wt in term_weights:
            if wt >= 0:
                d1.append((term, round(wt, 3)))
            else:
                d2.append((term, round(wt, 3)))

        print('Direction 1:', d1)
        print('-' * 50)
        print('Direction 2:', d2)
        print('-' * 50)
        print()


def run_lda():
    global lda_model, document_topics
    lda_model = LatentDirichletAllocation(n_components=TOTAL_TOPICS, max_iter=500,
                                          max_doc_update_iter=50,
                                          learning_method='online', batch_size=1740,
                                          learning_offset=50.,
                                          random_state=42, n_jobs=16)
    document_topics = lda_model.fit_transform(cv_train_features)


def display_lda_results():
    topic_terms = lda_model.components_
    topic_key_term_idxs = np.argsort(-np.absolute(topic_terms), axis=1)[:,
                          :top_terms]
    topic_keyterms = vocabulary[topic_key_term_idxs]
    topics = [', '.join(topic) for topic in topic_keyterms]
    # pd.set_option('display.max_colwidth', -1)
    topics_df = pd.DataFrame(topics,
                             columns=['Terms per Topic'],
                             index=['Topic' + str(t) for t in
                                    range(1, TOTAL_TOPICS + 1)])
    topics_df
    pd.options.display.float_format = '{:,.3f}'.format
    dt_df = pd.DataFrame(document_topics,
                         columns=['T' + str(i) for i in range(1, TOTAL_TOPICS + 1)])
    dt_df.T
    pd.options.display.float_format = '{:,.5f}'.format
    pd.set_option('display.max_colwidth', 200)

    max_contrib_topics = dt_df.max(axis=0)
    dominant_topics = max_contrib_topics.index
    contrib_perc = max_contrib_topics.values
    document_numbers = [dt_df[dt_df[t] == max_contrib_topics.loc[t]].index[0]
                        for t in dominant_topics]
    documents = [train_articles[i] for i in document_numbers]

    results_df = pd.DataFrame(
        {'Dominant Topic': dominant_topics, 'Contribution %': contrib_perc,
         'Paper Num': document_numbers, 'Topic': topics_df['Terms per Topic'],
         'Paper Name': documents})
    return topics_df, dt_df.T, results_df


def run_nmf():
    global nmf_model, document_topics
    nmf_model = NMF(n_components=TOTAL_TOPICS, solver='cd', max_iter=50000,
                    random_state=42, alpha=.1, l1_ratio=.85)
    document_topics = nmf_model.fit_transform(cv_train_features)


def display_nmf_results():
    topic_terms = nmf_model.components_
    topic_key_term_idxs = np.argsort(-np.absolute(topic_terms), axis=1)[:,
                          :top_terms]
    topic_keyterms = vocabulary[topic_key_term_idxs]
    topics = [', '.join(topic) for topic in topic_keyterms]
    topics_df = pd.DataFrame(topics,
                             columns=['Terms per Topic'],
                             index=['Topic' + str(t) for t in
                                    range(1, TOTAL_TOPICS + 1)])
    pd.options.display.float_format = '{:,.3f}'.format
    dt_df = pd.DataFrame(document_topics,
                         columns=['T' + str(i) for i in range(1, TOTAL_TOPICS + 1)])
    dt_df.head(10)
    pd.options.display.float_format = '{:,.5f}'.format
    pd.set_option('display.max_colwidth', 200)

    max_score_topics = dt_df.max(axis=0)
    dominant_topics = max_score_topics.index
    term_score = max_score_topics.values
    document_numbers = [dt_df[dt_df[t] == max_score_topics.loc[t]].index[0]
                        for t in dominant_topics]
    documents = [train_articles[i] for i in document_numbers]

    results_df = pd.DataFrame(
        {'Dominant Topic': dominant_topics, 'Max Score': term_score,
         'Paper Num': document_numbers, 'Topic': topics_df['Terms per Topic'],
         'Paper Name': documents})
    results_df
    return topics_df, dt_df.T, results_df


def create_prediction_results_df():
    global best_topics
    best_topics = [[(topic, round(sc, 3))
                    for topic, sc in sorted(enumerate(topic_predictions[i]),
                                            key=lambda row: -row[1])[:2]]
                   for i in range(len(topic_predictions))]

    prediction_results_df = pd.DataFrame()
    prediction_results_df['Papers'] = range(1, len(test_articles) + 1)
    prediction_results_df['Dominant Topics'] = [[topic_num + 1 for topic_num, sc in item] for
                                     item in best_topics]
    res = prediction_results_df.set_index(['Papers'])['Dominant Topics'].apply(
        pd.Series).stack().reset_index(level=1, drop=True)
    prediction_results_df = pd.DataFrame({'Dominant Topics': res.values}, index=res.index)
    prediction_results_df['Topic Score'] = [topic_sc for topic_list in
                                 [[round(sc * 100, 2)
                                   for topic_num, sc in item]
                                  for item in best_topics]
                                 for topic_sc in topic_list]

    prediction_results_df['Topic Desc'] = [topics_df.iloc[t - 1]['Terms per Topic'] for t in
                                prediction_results_df['Dominant Topics'].values]
    prediction_results_df['Paper Desc'] = [test_articles[i - 1][:200] for i in
                                prediction_results_df.index.values]

    return prediction_results_df


# run_lsi()
# display_lsi_results()

# run_lda()
# topics_df, dt_df, results_df = display_lda_results()
# topic_predictions = lda_model.transform(cv_test_features)

run_nmf()
topics_df, dt_df, results_df = display_nmf_results()
topic_predictions = nmf_model.transform(cv_test_features)

prediction_results_df = create_prediction_results_df()
predictions = np.array(prediction_results_df['Dominant Topics'])[::2]

correct_predictions = 0
for i in range(0, len(test_label_names)):
    if topic_by_order[predictions[i] - 1] == test_label_names[i]:
        correct_predictions += 1

acc = correct_predictions / len(test_label_names)

# # # Teraz przeprowadź to samo dla NMF
