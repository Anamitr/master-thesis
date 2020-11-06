import gensim
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import util
from topic_classification.experiment_config import CLASSIFIER_ITERATION, \
    CLASSIFIERS_AND_RESULTS_DIR_PATH


def document_vectorize(corpus, model, num_features):
    vocabulary = set(model.wv.index2word)

    def average_word_vectors(words, model, vocabulary, num_features):
        feature_vector = np.zeros((num_features,), dtype="float64")
        nwords = 0.

        for word in words:
            if word in vocabulary:
                nwords = nwords + 1.
                feature_vector = np.add(feature_vector, model.wv[word])
        if nwords:
            feature_vector = np.divide(feature_vector, nwords)

        return feature_vector

    features = [
        average_word_vectors(tokenized_sentence, model, vocabulary, num_features)
        for tokenized_sentence in corpus]
    return np.array(features)


def document_vectorize_with_fasttext_model(corpus, fasttext_model, num_features,
                                           vocabulary: set):
    def average_word_vectors(words, fasttext_model, vocabulary, num_features):
        feature_vector = np.zeros((num_features,), dtype="float64")
        nwords = 0.

        for word in words:
            if word in vocabulary:
                nwords = nwords + 1.
                feature_vector = np.add(feature_vector, fasttext_model.
                                        get_word_vector(word))
        if nwords:
            feature_vector = np.divide(feature_vector, nwords)

        return feature_vector

    features = [
        average_word_vectors(tokenized_sentence, fasttext_model, vocabulary,
                             num_features)
        for tokenized_sentence in corpus]
    return np.array(features)


def get_simple_bag_of_words_features(train_corpus, test_corpus):
    cv = CountVectorizer(binary=False, min_df=0.0, max_df=1.0)
    cv_train_features = cv.fit_transform(train_corpus)
    cv_test_features = cv.transform(test_corpus)
    print('BOW model:> Train features shape:', cv_train_features.shape,
          ' Test features shape:', cv_test_features.shape, '\n')
    return cv_train_features, cv_test_features


def get_tf_idf_features(train_corpus, test_corpus):
    tv = TfidfVectorizer(use_idf=True, min_df=0.0, max_df=1.0)
    tv_train_features = tv.fit_transform(train_corpus)
    tv_test_features = tv.transform(test_corpus)
    print('TFIDF model:> Train features shape:', tv_train_features.shape,
          ' Test features shape:', tv_test_features.shape)
    return tv_train_features, tv_test_features


def get_word2vec_features(tokenized_train: list, w2v_num_features: int):
    w2v_model = gensim.models.Word2Vec(sentences=tokenized_train,
                                       size=w2v_num_features,
                                       window=100, min_count=2, sample=1e-3, sg=1,
                                       iter=5, workers=10)
    util.save_object(w2v_model, CLASSIFIERS_AND_RESULTS_DIR_PATH + 'w2v_model' + str(
        CLASSIFIER_ITERATION) + '.pkl')
    return w2v_model
