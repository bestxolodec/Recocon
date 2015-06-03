import pickle
from bs4 import UnicodeDammit
import re
import requests
import lxml
import lxml.html
from lxml.html import etree
import chardet
from lxml import html

from analizer import Collection

with open('./links.dump', 'rb') as f:
    urls = [link[1] for link in pickle.load(f)]

c = Collection(urls)

texts = c.get_tokenized_texts()
dictionary = c.get_dictionary()
corpus = c.get_corpus()

# gensim starts
from gensim import corpora, models, similarities
tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

n_topics = 10
lda = models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=n_topics)
# most probable words inside every topic
topn = 5
for i in range(0, n_topics):
    temp = lda.show_topic(i, topn)
    terms = []
    for term in temp:
        terms.append(term[1])
    print "Top 5 terms for topic #" + str(i) + "  "+ ", ".join(terms)
    print "Top 5 terms for topic #{}: {}".format(i, ", ".join(terms))
