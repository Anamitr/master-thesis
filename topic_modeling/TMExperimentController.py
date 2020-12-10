import os

from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

from topic_classification.constants import Dataset, ModelingMethod
from topic_classification.dataset_utils import load_20newsgroups, \
    load_preprocessed_news_category_dataset, load_preprocessed_bbc_news_summary, \
    load_preprocessed_arxiv_metadata_dataset


class TMExperimentController:
    def __init__(self):
        # os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'
        # Constants
        self.topics_in_order = None
        # Paths
        self.BASE_TOPIC_CLASSIFICATION_DIR_PATH = '/home/konrad/Repositories/' \
                                                  'master-thesis/topic_classification/'
        self.TOPIC_CLASSIFICATION_DATA_PATH = self.BASE_TOPIC_CLASSIFICATION_DIR_PATH + \
                                              'topic_class_data/'
        self.TRAIN_DATA_FOR_FASTTEXT_PATH = None
        self.TEST_DATA_FOR_FASTTEXT_PATH = None
        # Constant parameters
        self.TOTAL_TOPICS = None
        self.NUM_OF_TOP_TERMS = 20
        # Models
        self.lsi_model = None
        self.lda_model = None
        self.nmf_model = None
        self.modeling_model = None
        self.document_topics = None
        # Experiment specific
        self.dataset_enum = None
        self.modeling_method_enum = None
        # Dataset
        self.data_df = None
        self.avg_dataset_length = None
        self.train_corpus = None
        self.test_corpus = None
        self.train_label_names = None
        self.test_label_names = None
        self.tokenized_train = None
        self.tokenized_test = None
        self.vocabulary = None
        self.train_articles = None
        self.test_articles = None
        # Feature Extraction
        self.cv = None
        self.train_features = None
        self.test_features = None
        # Modeling results
        self.topics_df = None
        self.dt_df = None
        self.modeling_results_df = None
        # Classification results
        self.topic_predictions = None
        self.prediction_results_df = None
        self.predictions = None
        self.correct_predictions = None
        self.acc = None

    def set_variables(self, dateset_enum, modeling_method_enum, num_of_topics):
        self.dataset_enum = dateset_enum
        self.TRAIN_DATA_FOR_FASTTEXT_PATH = self.TOPIC_CLASSIFICATION_DATA_PATH + \
                                            self.dataset_enum.name + \
                                            '_fasttext_train_formatted.txt'
        self.TEST_DATA_FOR_FASTTEXT_PATH = self.TOPIC_CLASSIFICATION_DATA_PATH + \
                                           self.dataset_enum.name + \
                                           '_fasttext_test_formatted.txt'
        self.TOTAL_TOPICS = num_of_topics
        self.modeling_method_enum = modeling_method_enum
        # Init models
        self.lsi_model = TruncatedSVD(n_components=self.TOTAL_TOPICS, n_iter=500,
                                      random_state=42)
        self.lda_model = LatentDirichletAllocation(n_components=self.TOTAL_TOPICS,
                                                   max_iter=11,
                                                   max_doc_update_iter=50,
                                                   learning_method='online',
                                                   batch_size=1740,
                                                   learning_offset=50.,
                                                   random_state=42, n_jobs=16,
                                                   verbose=1)
        self.nmf_model = NMF(n_components=self.TOTAL_TOPICS, solver='cd',
                             max_iter=50000,
                             random_state=42, alpha=.1, l1_ratio=.85)

    def run_experiment(self):
        # Load dataset
        self.data_df = self.get_dataset_from_name(self.dataset_enum)
        print('Got dataset:', self.dataset_enum)

        self.train_corpus, self.test_corpus, self.train_label_names, \
        self.test_label_names = train_test_split(
            np.array(self.data_df['Clean Article']),
            np.array(self.data_df['Target Name']),
            test_size=0.33, random_state=42)

        self.train_articles = [art_str.split(' ') for art_str in self.train_corpus]
        self.test_articles = [art_str.split(' ') for art_str in self.test_corpus]

        # # # Feature Extraction
        self.cv = CountVectorizer(min_df=20, max_df=0.6, ngram_range=(1, 2),
                                  token_pattern=None, tokenizer=lambda doc: doc,
                                  preprocessor=lambda doc: doc)
        self.train_features = self.cv.fit_transform(self.train_articles)
        self.test_features = self.cv.transform(self.test_articles)
        self.vocabulary = np.array(self.cv.get_feature_names())
        print('Extracted features')
        # # #
        self.run_modeling()
        print('Finished modeling')
        self.topics_df, self.dt_df, self.modeling_results_df = \
            self.generate_modeling_results()
        print('Modeling results ready')

    def get_dataset_from_name(self, dataset_name):
        dataset_switcher = {
            Dataset.ds20newsgroups: load_20newsgroups,
            Dataset.news_category_dataset:
                load_preprocessed_news_category_dataset,
            Dataset.bbc_news_summary: load_preprocessed_bbc_news_summary,
            Dataset.arxiv_metadata: load_preprocessed_arxiv_metadata_dataset,
        }
        return dataset_switcher.get(dataset_name, lambda: "Invalid dataset")(
            self.TOPIC_CLASSIFICATION_DATA_PATH
        )

    def run_modeling(self):
        modeling_method_switcher = {
            ModelingMethod.LSI: self.lsi_model,
            ModelingMethod.LDA: self.lda_model,
            ModelingMethod.NMF: self.nmf_model
        }
        self.modeling_model = modeling_method_switcher.get(self.modeling_method_enum)
        self.document_topics = self.modeling_model.fit_transform(self.train_features)
        pass

    def generate_modeling_results(self):
        topic_terms = self.modeling_model.components_
        topic_key_term_idxs = np.argsort(-np.absolute(topic_terms), axis=1)[:,
                              :self.NUM_OF_TOP_TERMS]
        topic_keyterms = self.vocabulary[topic_key_term_idxs]
        topics = [', '.join(topic) for topic in topic_keyterms]
        # pd.set_option('display.max_colwidth', -1)
        topics_df = pd.DataFrame(topics,
                                 columns=['Terms per Topic'],
                                 index=['Topic' + str(t) for t in
                                        range(1, self.TOTAL_TOPICS + 1)])
        # topics_df
        pd.options.display.float_format = '{:,.3f}'.format
        dt_df = pd.DataFrame(self.document_topics,
                             columns=['T' + str(i) for i in
                                      range(1, self.TOTAL_TOPICS + 1)])
        # dt_df.T
        pd.options.display.float_format = '{:,.5f}'.format
        pd.set_option('display.max_colwidth', 200)

        max_contrib_topics = dt_df.max(axis=0)
        dominant_topics = max_contrib_topics.index
        contrib_perc = max_contrib_topics.values
        document_numbers = [dt_df[dt_df[t] == max_contrib_topics.loc[t]].index[0]
                            for t in dominant_topics]
        documents = [self.train_articles[i] for i in document_numbers]

        results_df = pd.DataFrame(
            {'Dominant Topic': dominant_topics, 'Contribution %': contrib_perc,
             'Paper Num': document_numbers, 'Topic': topics_df['Terms per Topic'],
             'Paper Name': documents})
        return topics_df, dt_df.T, results_df

    def set_topics_in_order(self, topics_in_order):
        self.topics_in_order = topics_in_order

    def test_prediction_accuracy(self):
        if self.topics_in_order is None:
            print('Set topics_by_order')

        self.topic_predictions = self.modeling_model.transform(self.test_features)
        self.prediction_results_df = self.create_prediction_results_df()
        self.predictions = np.array(self.prediction_results_df['Dominant Topics'])[::2]

        self.correct_predictions = 0
        for i in range(0, len(self.test_label_names)):
            if self.topics_in_order[self.predictions[i] - 1] == self.test_label_names[i]:
                self.correct_predictions += 1

        self.acc = self.correct_predictions / len(self.test_label_names)
        print('Accuracy =', self.acc)

    def create_prediction_results_df(self):
        global best_topics
        best_topics = [[(topic, round(sc, 3))
                        for topic, sc in sorted(enumerate(self.topic_predictions[i]),
                                                key=lambda row: -row[1])[:2]]
                       for i in range(len(self.topic_predictions))]

        prediction_results_df = pd.DataFrame()
        prediction_results_df['Papers'] = range(1, len(self.test_articles) + 1)
        prediction_results_df['Dominant Topics'] = [
            [topic_num + 1 for topic_num, sc in item] for
            item in best_topics]
        res = prediction_results_df.set_index(['Papers'])['Dominant Topics'].apply(
            pd.Series).stack().reset_index(level=1, drop=True)
        prediction_results_df = pd.DataFrame({'Dominant Topics': res.values},
                                             index=res.index)
        prediction_results_df['Topic Score'] = [topic_sc for topic_list in
                                                [[round(sc * 100, 2)
                                                  for topic_num, sc in item]
                                                 for item in best_topics]
                                                for topic_sc in topic_list]

        prediction_results_df['Topic Desc'] = [
            self.topics_df.iloc[t - 1]['Terms per Topic'] for t in
            prediction_results_df['Dominant Topics'].values]
        prediction_results_df['Paper Desc'] = [self.test_articles[i - 1][:200] for
                                               i in
                                               prediction_results_df.index.values]

        return prediction_results_df
