import re
import urllib.request

import bs4 as bs
import nltk
import heapq

from rouge_score import rouge_scorer

from nlp_processing import find_weighted_frequencies, calculate_sentences_score
from text_format_util import get_text_from_wikipedia, read_book_and_format


def get_tsavo_man_eaters_wiki_summary():
    raw_text, formatted_text = get_text_from_wikipedia("https://en.wikipedia.org/wiki/Tsavo_Man-Eaters")

    print("Tsavo text:\n", raw_text)

    # Converting Text To Sentences
    sentence_list = nltk.sent_tokenize(raw_text)

    # Find Weighted Frequency of Occurrence
    word_frequencies = find_weighted_frequencies(formatted_text)

    # Calculating Sentence Scores
    sentence_scores = calculate_sentences_score(sentence_list, word_frequencies)

    # Getting the Summary
    summary_sentences = heapq.nlargest(7, sentence_scores, key=sentence_scores.get)

    summary = ' '.join(summary_sentences)
    return summary, raw_text, formatted_text


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
    return summary, raw_text, formatted_text


tsavo_summary, tsavo_raw_text, tsavo_formatted_text = get_tsavo_man_eaters_wiki_summary()
# print("Summary:\n", tsavo_summary)
egypt_summary, egypt_raw_text, egypt_formatted_text = get_egyptian_history_book_summary()

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
tsavo_scores = scorer.score(tsavo_raw_text, tsavo_summary)
egypt_scores = scorer.score(egypt_raw_text, egypt_summary)

print("tsavo scores:\n", tsavo_scores)
print("egypt scores:\n", egypt_scores)
# test_gensim()
