import os

import fasttext
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import util
from topic_classification.classifiers_definitions_utils import \
    multinominal_naive_bayes_classifier, logistic_regression_classifier, \
    support_vector_machines_classifier, \
    svm_with_stochastic_gradient_descent_classifier, random_forest_classifier, \
    gradient_boosting_machines_classifier
from topic_classification.constants import Dataset, FeatureExtractionMethod, \
    ClassificationMethod
from topic_classification.dataset_utils import load_20newsgroups, \
    load_preprocessed_news_category_dataset, load_preprocessed_bbc_news_summary, \
    load_preprocessed_arxiv_metadata_dataset, get_dataset_avg_length
import text_preprocessing.text_normalizer as tn
from topic_classification.datastructures import TrainingData
from topic_classification.display_utils import create_2_bar_plot, create_bar_plot
from topic_classification.feature_extraction_utils import \
    get_simple_bag_of_words_features, get_tf_idf_features, \
    get_word2vec_trained_model, document_vectorize_with_fasttext_model, \
    document_vectorize, train_fasttext_model
from topic_classification.train_utils import get_chosen_classifiers, \
    train_multiple_classifiers


def get_classifier_tuple(classifier_enum):
    classifier_switcher = {
        ClassificationMethod.Naive_Bayes_Classifier:
            multinominal_naive_bayes_classifier,
        ClassificationMethod.Logistic_Regression:
            logistic_regression_classifier,
        ClassificationMethod.Support_Vector_Machines:
            support_vector_machines_classifier,
        ClassificationMethod.SVM_with_SGD:
            svm_with_stochastic_gradient_descent_classifier,
        ClassificationMethod.Random_Forest:
            random_forest_classifier,
        ClassificationMethod.Gradient_Boosting_Machines:
            gradient_boosting_machines_classifier,
    }
    return classifier_switcher.get(classifier_enum, lambda: "Invalid dataset")


class ExperimentController:
    def __init__(self, exp_name: str = 'Unknown', classifier_iter='-1') -> None:
        self.exp_name = exp_name
        self.classifier_iter = classifier_iter
        # Paths
        self.BASE_TOPIC_CLASSIFICATION_DIR_PATH = '/home/konrad/Repositories/' \
                                                  'master-thesis/topic_classification/'
        self.TOPIC_CLASSIFICATION_DATA_PATH = self.BASE_TOPIC_CLASSIFICATION_DIR_PATH + \
                                              'topic_class_data/'
        self.CLASSIFIERS_AND_RESULTS_DIR_PATH = self.BASE_TOPIC_CLASSIFICATION_DIR_PATH \
                                                + 'trained_classifiers/' \
                                                + self.exp_name + '/'
        self.RESULTS_PATH = self.CLASSIFIERS_AND_RESULTS_DIR_PATH + 'results_' + \
                            str(self.classifier_iter) + '.pkl'
        self.WORD2VEC_MODEL_SAVE_PATH = self.CLASSIFIERS_AND_RESULTS_DIR_PATH + \
                                        'w2v_model_' + str(
            self.classifier_iter) + '.pkl'
        self.TRAIN_DATA_FOR_FASTTEXT_PATH = None
        self.TEST_DATA_FOR_FASTTEXT_PATH = None
        self.FAST_TEXT_SAVE_PATH = self.CLASSIFIERS_AND_RESULTS_DIR_PATH + \
                                   'fasttext_model_' + \
                                   str(self.classifier_iter) + '.pkl'

        if not os.path.exists(self.CLASSIFIERS_AND_RESULTS_DIR_PATH):
            os.makedirs(self.CLASSIFIERS_AND_RESULTS_DIR_PATH)

        # Experiment specific
        self.dataset_enum = None
        self.feature_extraction_method = None
        self.classifiers = None
        # Config
        self.TEST_SET_SIZE_RATIO = 0.33
        self.NUM_OF_VEC_FEATURES = 1000
        self.SCORE_DECIMAL_PLACES = 4
        self.TIME_DECIMAL_PLACES = 2
        # Flags
        self.should_load_embedding_model = True
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
        # Feature extraction
        self.embedding_model = None
        self.train_features = None
        self.test_features = None
        self.training_data = None
        # Classifiers
        self.classifier_list = None
        self.classifier_name_list = None
        self.classifier_name_shortcut_list = None
        # Results
        self.results = None
        self.cv_mean_scores = None
        self.test_scores = None
        self.elapsed_times = None
        super().__init__()

    def set_variables(self, dataset_enum, feature_extraction_method, classifiers,
                      should_load_embedding_model=True):
        # Set variables
        self.dataset_enum = dataset_enum
        self.TRAIN_DATA_FOR_FASTTEXT_PATH = self.TOPIC_CLASSIFICATION_DATA_PATH + \
                                            self.dataset_enum.name + \
                                            '_fasttext_train_formatted.txt'
        self.TEST_DATA_FOR_FASTTEXT_PATH = self.TOPIC_CLASSIFICATION_DATA_PATH + \
                                           self.dataset_enum.name + \
                                           '_fasttext_test_formatted.txt'
        self.feature_extraction_method = feature_extraction_method
        self.classifiers = classifiers
        self.should_load_embedding_model = should_load_embedding_model
        self.classifier_list, self.classifier_name_list, \
        self.classifier_name_shortcut_list = \
            self.get_chosen_classifiers_and_their_metadata()

    def run_experiment(self):
        print('Running', self.exp_name, ', it.', self.classifier_iter)
        # Load dataset
        self.data_df = self.get_dataset_from_name(self.dataset_enum)
        self.avg_dataset_length = get_dataset_avg_length(self.data_df)
        print('Got dataset:', self.dataset_enum, 'num of cat.',
              self.get_num_of_categories())
        # Split on train and test dataset
        self.train_corpus, self.test_corpus, self.train_label_names, \
        self.test_label_names = \
            train_test_split(np.array(self.data_df['Clean Article']),
                             np.array(self.data_df['Target Name']),
                             test_size=self.TEST_SET_SIZE_RATIO, random_state=42)
        # Tokenize corpus
        self.tokenized_train = [tn.tokenizer.tokenize(text) for text in
                                self.train_corpus]
        self.tokenized_test = [tn.tokenizer.tokenize(text) for text in
                               self.test_corpus]
        # Get list of words (I know, cool, not readable one-liner)
        data_word_list = ''.join(list(self.data_df['Clean Article'])).split(' ')
        self.vocabulary = set(data_word_list)
        # Calculate features
        self.train_features, self.test_features = self.get_features()
        print('Train features shape:', self.train_features.shape,
              ' Test features shape:', self.test_features.shape)
        # Pack data in one class
        self.training_data = TrainingData(self.train_features,
                                          self.train_label_names,
                                          self.test_features, self.test_label_names)
        # Perform actual training
        self.results = train_multiple_classifiers(self.classifier_list,
                                                  self.classifier_name_list,
                                                  self.training_data,
                                                  self.
                                                  CLASSIFIERS_AND_RESULTS_DIR_PATH,
                                                  self.classifier_iter,
                                                  self.RESULTS_PATH)
        # Extract scores for plotting
        self.display_results()
        pass

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

    def get_features(self):
        print('Get features:', self.feature_extraction_method)
        if self.feature_extraction_method == FeatureExtractionMethod.BOW:
            return get_simple_bag_of_words_features(self.train_corpus,
                                                    self.test_corpus)
        elif self.feature_extraction_method == FeatureExtractionMethod.TF_IDF:
            return get_tf_idf_features(self.train_corpus, self.test_corpus)
        elif self.feature_extraction_method == FeatureExtractionMethod.WORD2VEC:
            if self.should_load_embedding_model:
                print('Loading embedding model from disk')
                self.embedding_model = util.load_object(
                    self.WORD2VEC_MODEL_SAVE_PATH)
            else:
                print('Calculating embeddings')
                self.embedding_model = get_word2vec_trained_model(
                    self.tokenized_test, self.NUM_OF_VEC_FEATURES)
                util.save_object(self.embedding_model,
                                 self.CLASSIFIERS_AND_RESULTS_DIR_PATH + 'w2v_model_'
                                 + str(self.classifier_iter) + '.pkl')
            return self.get_document_embeddings_from_word2vec()
        elif self.feature_extraction_method == FeatureExtractionMethod.FASTTEXT:
            if self.should_load_embedding_model:
                print('Loading embedding model from disk')
                self.embedding_model = fasttext.load_model(self.FAST_TEXT_SAVE_PATH)
            else:
                print('Calculating embeddings')
                if not os.path.exists(self.TRAIN_DATA_FOR_FASTTEXT_PATH):
                    self.reformat_and_save_data_for_fasttext()
                self.embedding_model = train_fasttext_model(
                    self.TRAIN_DATA_FOR_FASTTEXT_PATH,
                    self.NUM_OF_VEC_FEATURES, epoch=100)
                self.embedding_model.save_model(self.FAST_TEXT_SAVE_PATH)
            return self.get_document_embeddings_from_fasttext()
        else:
            print('No such feature extraction method:',
                  self.feature_extraction_method)

    def get_document_embeddings_from_word2vec(self):
        train_features = document_vectorize(corpus=self.tokenized_train,
                                            model=self.embedding_model,
                                            num_features=
                                            self.NUM_OF_VEC_FEATURES)
        test_features = document_vectorize(corpus=self.tokenized_test,
                                           model=self.embedding_model,
                                           num_features=
                                           self.NUM_OF_VEC_FEATURES)
        return train_features, test_features

    def get_document_embeddings_from_fasttext(self):
        train_features = document_vectorize_with_fasttext_model(
            corpus=self.tokenized_train,
            fasttext_model=self.embedding_model,
            num_features=self.NUM_OF_VEC_FEATURES,
            vocabulary=self.vocabulary)
        test_features = document_vectorize_with_fasttext_model(
            corpus=self.tokenized_test,
            fasttext_model=self.embedding_model,
            num_features=self.NUM_OF_VEC_FEATURES,
            vocabulary=self.vocabulary)
        return train_features, test_features

    def get_chosen_classifiers_and_their_metadata(self):
        classifiers_tuples = []
        for classifier_enum in self.classifiers:
            classifiers_tuples.append(get_classifier_tuple(classifier_enum))

        return get_chosen_classifiers(classifiers_tuples)

    def print_results_table(self):
        results_df = pd.DataFrame([[round(item[1], self.SCORE_DECIMAL_PLACES),
                                    round(item[2], self.SCORE_DECIMAL_PLACES)
                                    ] for item in self.results]).T
        results_df.columns = self.classifier_name_shortcut_list
        results_df.index = ['CV Mean', 'Test accuracy']
        results_df.to_clipboard()

    def reformat_and_save_data_for_fasttext(self):
        ft_train_data_formatted = ''
        for i in range(0, len(self.train_corpus)):
            if i in self.data_df.index:
                ft_train_data_formatted += '__label__' + self.train_label_names[i] + \
                                           ' ' + self.train_corpus[i] + '\n'
        util.save_object(ft_train_data_formatted, self.TRAIN_DATA_FOR_FASTTEXT_PATH)

        ft_test_data_formatted = ''
        for i in range(0, len(self.test_corpus)):
            if i in self.data_df.index:
                ft_test_data_formatted += '__label__' + self.test_label_names[
                    i] + ' ' + self.test_corpus[i] + '\n'
        util.save_object(ft_test_data_formatted, self.TEST_DATA_FOR_FASTTEXT_PATH)

    def load_results_from_disk(self):
        results_list = []
        for i in range(0, len(self.classifier_list)):
            results = util.load_object(self.CLASSIFIERS_AND_RESULTS_DIR_PATH +
                                       util.convert_name_to_filename(
                                           self.classifier_name_list[
                                               i]) + '_' + self.classifier_iter
                                       + '_results.pkl')
            results_list.append(results)
        self.results = results_list
        return results_list

    def display_results(self):
        # Extract scores for plotting
        self.cv_mean_scores = [round(result[1], self.SCORE_DECIMAL_PLACES) for result
                               in self.results]
        self.test_scores = [round(result[2], self.SCORE_DECIMAL_PLACES) for result in
                            self.results]
        self.elapsed_times = [round(result[3], self.TIME_DECIMAL_PLACES) for result
                              in self.results]
        create_2_bar_plot(self.classifier_name_shortcut_list, 'Classifier scores',
                          'Accuracy',
                          self.cv_mean_scores, self.test_scores, 'cv means',
                          'test set',
                          y_range_tuple=(0, 1), should_autolabel=True)
        create_bar_plot(self.classifier_name_shortcut_list, 'Elapsed training times',
                        'Time in seconds', self.elapsed_times, color='red')
        self.print_results_table()

    def reload_base_path(self, base_path: str):
        self.BASE_TOPIC_CLASSIFICATION_DIR_PATH = base_path
        self.TOPIC_CLASSIFICATION_DATA_PATH = self.BASE_TOPIC_CLASSIFICATION_DIR_PATH + \
                                              'topic_class_data/'
        self.CLASSIFIERS_AND_RESULTS_DIR_PATH = self.BASE_TOPIC_CLASSIFICATION_DIR_PATH \
                                                + 'trained_classifiers/' \
                                                + self.exp_name + '/'
        self.RESULTS_PATH = self.CLASSIFIERS_AND_RESULTS_DIR_PATH + 'results_' + \
                            str(self.classifier_iter) + '.pkl'
        self.WORD2VEC_MODEL_SAVE_PATH = self.CLASSIFIERS_AND_RESULTS_DIR_PATH + \
                                        'w2v_model_' + str(
            self.classifier_iter) + '.pkl'
        self.TRAIN_DATA_FOR_FASTTEXT_PATH = self.TOPIC_CLASSIFICATION_DATA_PATH + \
                                            self.dataset_enum.name + \
                                            '_fasttext_train_formatted.txt'
        self.TEST_DATA_FOR_FASTTEXT_PATH = self.TOPIC_CLASSIFICATION_DATA_PATH + \
                                           self.dataset_enum.name + \
                                           '_fasttext_test_formatted.txt'
        self.FAST_TEXT_SAVE_PATH = self.CLASSIFIERS_AND_RESULTS_DIR_PATH + \
                                   'fasttext_model_' + \
                                   str(self.classifier_iter) + '.pkl'

    def get_cv_mean_scores(self):
        return [round(result[1], 4) for result in self.results]

    def get_test_scores(self):
        return [round(result[2], 4) for result in self.results]

    def get_num_of_categories(self):
        return len(set(self.data_df['Target Name']))

