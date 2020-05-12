import re
import urllib.request

import bs4 as bs


def read_book_and_format(path: str, begin_line: str = None, end_line: str = None):
    with open(path, 'r') as file:
        data = file.read()
    if begin_line:
        data = data.split(begin_line)[1]
    if end_line:
        data = data.split(end_line)[0]

    # Removing Square Brackets and Extra Spaces
    raw_text = re.sub(r'\[[0-9]*\]', ' ', data)
    raw_text = re.sub(r'\s+', ' ', raw_text)

    # Removing special characters and digits
    formatted_text = re.sub('[^a-zA-Z]', ' ', raw_text)
    formatted_text = re.sub(r'\s+', ' ', formatted_text)

    return raw_text, formatted_text


def get_text_from_wikipedia(url: str):
    scraped_data = urllib.request.urlopen(url)
    article = scraped_data.read()

    parsed_article = bs.BeautifulSoup(article, 'lxml')

    paragraphs = parsed_article.find_all('p')

    article_text = ""

    for p in paragraphs:
        article_text += p.text

    # Removing Square Brackets and Extra Spaces
    article_text = re.sub(r'\[[0-9]*\]', ' ', article_text)
    article_text = re.sub(r'\s+', ' ', article_text)

    # Removing special characters and digits
    formatted_article_text = re.sub('[^a-zA-Z]', ' ', article_text)
    formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)

    return article_text, formatted_article_text
