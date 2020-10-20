from gensim import models, corpora, similarities

from topic_modelling_util import clean_text, get_data

NUM_TOPICS = 10

# def test_gensim():
data = get_data()
# For gensim we need to tokenize the data and filter out stopwords
tokenized_data = []
for text in data:
    tokenized_data.append(clean_text(text))

# Build a Dictionary - association word to numeric id
dictionary = corpora.Dictionary(tokenized_data)

# Transform the collection of texts to a numerical form
corpus = [dictionary.doc2bow(text) for text in tokenized_data]

# Have a look at how the 20th document looks like: [(word_id, count), ...]
print(corpus[20])
# [(12, 3), (14, 1), (21, 1), (25, 5), (30, 2), (31, 5), (33, 1), (42, 1), (43, 2),  ...

# Build the LDA model
lda_model = models.LdaModel(corpus=corpus, num_topics=NUM_TOPICS, id2word=dictionary)

# Build the LSI model
lsi_model = models.LsiModel(corpus=corpus, num_topics=NUM_TOPICS, id2word=dictionary)

print("LDA Model:")

for idx in range(NUM_TOPICS):
    # Print the first 10 most representative topics
    print("Topic #%s:" % idx, lda_model.print_topic(idx, 10))

print("=" * 20)

print("LSI Model:")

for idx in range(NUM_TOPICS):
    # Print the first 10 most representative topics
    print("Topic #%s:" % idx, lsi_model.print_topic(idx, 10))

print("=" * 20)

text = "The economy is working better than ever"
bow = dictionary.doc2bow(clean_text(text))

print(lsi_model[bow])
# [(0, 0.091615426138426506), (1, -0.0085557463300508351), (2, 0.016744863677828108), (3, 0.040508186718598529), (4, 0.014201267714185898), (5, -0.012208538275305329), (6, 0.031254053085582149), (7, 0.017529584659403553), (8, 0.056957633371540077),
(9, 0.025989149894888153)

print(lda_model[bow])

lda_index = similarities.MatrixSimilarity(lda_model[corpus])

# Let's perform some queries
similarities = lda_index[lda_model[bow]]
# Sort the similarities
similarities = sorted(enumerate(similarities), key=lambda item: -item[1])

# Top most similar documents:
print(similarities[:10])
# [(104, 0.87591344), (178, 0.86124849), (31, 0.8604598), (77, 0.84932965), (85, 0.84843522), (135, 0.84421808), (215, 0.84184396), (353, 0.84038532), (254, 0.83498049), (13, 0.82832891)]

# Let's see what's the most similar document
document_id, similarity = similarities[0]
print(data[document_id][:1000])
