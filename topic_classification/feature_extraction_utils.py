import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


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
    # # build BOW features on train articles
    cv = CountVectorizer(binary=False, min_df=0.0, max_df=1.0)
    cv_train_features = cv.fit_transform(train_corpus)
    # # transform test articles into features
    cv_test_features = cv.transform(test_corpus)
    print('BOW model:> Train features shape:', cv_train_features.shape,
          ' Test features shape:', cv_test_features.shape, '\n')
