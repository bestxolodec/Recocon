import pickle
from bs4 import UnicodeDammit
import re
import requests
import lxml
import lxml.html
from lxml.html import etree
import chardet
from lxml import html
from gensim import  corpora, models

from analizer import Collection
from analizer import  Page

# with open('./links.dump', 'rb') as f:
with open('150_links.dump', 'rb') as f:
        urls = [link[1] for link in pickle.load(f)]

c =  Collection(urls)
texts = c.get_tokenized_texts()
text = texts[1]
lda = models.LdaModel.load("../models/lda_on_bow")
all_texts = [word for text in texts for word in text]
id2word = corpora.Dictionary.load_from_text("../models/_wordids_stripped.txt")
corpus = [id2word.doc2bow(text) for text in texts]
all_corpus = id2word.doc2bow(all_texts)
wiki_corpus = corpora.MmCorpus("../models/_bow.mm")

