import re
import urllib.request

import bs4 as bs
import nltk
import heapq

from nlp_processing import find_weighted_frequencies, calculate_sentences_score
from text_format_util import get_text_from_wikipedia, read_book_and_format


def get_tsavo_man_eaters_wiki_summary():
    raw_text, formatted_text = get_text_from_wikipedia("https://en.wikipedia.org/wiki/Tsavo_Man-Eaters")

    # Converting Text To Sentences
    sentence_list = nltk.sent_tokenize(raw_text)

    # Find Weighted Frequency of Occurrence
    word_frequencies = find_weighted_frequencies(formatted_text)

    # Calculating Sentence Scores
    sentence_scores = calculate_sentences_score(sentence_list, word_frequencies)

    # Getting the Summary
    summary_sentences = heapq.nlargest(7, sentence_scores, key=sentence_scores.get)

    summary = ' '.join(summary_sentences)
    return summary


def get_egyptian_history_book_summary():
    raw_text, formatted_text = read_book_and_format("egyptian-history.txt")

    # Converting Text To Sentences
    sentence_list = nltk.sent_tokenize(raw_text)

    # Find Weighted Frequency of Occurrence
    word_frequencies = find_weighted_frequencies(formatted_text)

    # Calculating Sentence Scores
    sentence_scores = calculate_sentences_score(sentence_list, word_frequencies)

    # Getting the Summary
    summary_sentences = heapq.nlargest(7, sentence_scores, key=sentence_scores.get)

    summary = ' '.join(summary_sentences)
    return summary


# summary = get_tsavo_man_eaters_wiki_summary()
# summary = get_egyptian_history_book_summary()
# print(summary)

test_gensim()
