import re

from nltk import word_tokenize
from nltk.corpus import brown
from nltk.corpus import stopwords

STOPWORDS = stopwords.words('english')


def clean_text(text):
    tokenized_text = word_tokenize(text.lower())
    cleaned_text = [t for t in tokenized_text if t not in STOPWORDS and re.match('[a-zA-Z\-][a-zA-Z\-]{2,}', t)]
    return cleaned_text


def get_data():
    data = []

    for fileid in brown.fileids():
        document = ' '.join(brown.words(fileid))
        data.append(document)

    NO_DOCUMENTS = len(data)
    print(NO_DOCUMENTS)
    print(data[:5])
    return data
