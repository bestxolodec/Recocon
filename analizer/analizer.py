#!/usr/bin/env python
# encoding: utf-8

from lxml import html
from lxml.html import etree
import requests
import re
from nltk.stem import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from logger import Logger
import pickle
import os
from gensim import corpora
# , models, similarities
from bs4 import UnicodeDammit
import chardet


# if less then this threshold then use encoding, declared in html
# in case that chardet has commited an error in auto detection of charset
THRESHOLD_OF_CHARDETECT = 0.7


class ParsingError(Exception):
    def __init__(self, message):
        # Call the base class constructor with the parameters it needs
        super(ParsingError, self).__init__(message)


class Token(Logger):

    """ Class to hide stem of a word features under the hood. """

    def __init__(self, string, stem=None, stemmer=None):
        """ Initialize stem for Token, which is totally defined by string
        passed.

        Args:
            string (str): string, which represents a token
            stem (str): reduced form of a word `string`
        """
        self.string = string
        self.stem = stem or self._get_stem(string)
        self.stemmer = stemmer or SnowballStemmer('lang')

    def __eq__(self, other):
        return self.stem == other.stem

    def __repr__(self):
        return repr(self.string)

    def _get_stem(self, word, lang="russian"):
        """ Find stem of a word.

        Args:
            word (str): word to find stem of
            lang (str): language of that word. This could be one of the
                following: danish, dutch, english, finnish, french,
                german, hungarian, italian, norwegian, porter, portuguese,
                romanian, russian, spanish, swedish.

        Returns:
            stem  of the word `word`
        """
        return self.stemmer.stem(word)


class Page(Logger):

    """
    This class is designed to store all information about particular link,
    grab and clear data from such a link.
    We undoubdetly assume that page downloaded from a link entirely
    fits into memory.
    """

    # init all initial variables
    rate = None
    url = None
    descr = None
    tokenizer = None
    html = None
    text = None
    tokens = None

    def __init__(self, link, engine="google", tokenizer=None):
        """
        Args:
            link: typle of (rate, url, description) of a link
            engine:  engine this link comes from
            tokenizer: class that have method `tokenize` and could be called
                the following way: `tokenizer.tokenize(text)`, where text is
                string to be tokenized.
        """
        self.log.debug("{!r}".format(link))
        assert isinstance(link, tuple), "Wrong link format!"
        assert len(link) == 3, "Improper length of link tuple!"
        self.rate, self.url, self.descr = link
        self.tokenizer = tokenizer or self._get_tokenizer()

    def get_list_of_tokens(self):
        """ This function retrieves text from url and prepares it for
        further processing: clean from html tags, nonalpha characters,
        stopwords.

        Returns:
            list of tokens in a sequence they have appeared in text
        """
        if not self.tokens:
            assert self.url, "Wrong url provided!"
            self.text = self._get_text(remove_newlines=True)
            self.log.debug(u"First 100 characters of text from {url}: {text}"
                           "".format(url=self.url,
                                     text=self.text[:100]))
            self.tokens = self._get_tokens()
            self.log.debug(u"First 100 tokens of text from {url}: {tokens}"
                           "".format(url=self.url,
                                     tokens=self.tokens))
        return self.tokens

    def _get_text(self, remove_newlines=False):
        """ Retrieves html with provided url and parses it to fully remove
        all html tags, style declarations and scripts.

        Args:
            remove_newlines (bool): wheter perform cleaning of a \n\r
                sequencies or not.

        Returns:
            unicode object of the whole text without html tags

        """
        if not self.text:
            url = self.url
            try:
                self.log.debug("Try to get content from page {}".format(url))
                r = requests.get(url)
            except requests.exceptions.RequestException as e:
                self.log.warn("Unable to get page content of the url: {url}. "
                              "The reason: {exc!r}".format(url=url, exc=e))
                raise ParsingError(e.message)

            ud = UnicodeDammit(r.content, is_html=True)

            enc = ud.original_encoding.lower()
            declared_enc = ud.declared_html_encoding
            if declared_enc:
                declared_enc = declared_enc.lower()
            # possible misregocnition of an encoding
            if (declared_enc and enc != declared_enc):
                detect_dict = chardet.detect(r.content)
                det_conf = detect_dict["confidence"]
                det_enc = detect_dict["encoding"].lower()
                if enc == det_enc and det_conf < THRESHOLD_OF_CHARDETECT:
                    enc = declared_enc
            print "CHOOSED ENCODING: {}".format(enc)
            # if page contains any characters that differ from the main
            # encodin we will ignore them
            content = r.content.decode(enc, "ignore").encode(enc)
            htmlparser = etree.HTMLParser(encoding=enc)
            root = etree.HTML(content, parser=htmlparser)
            etree.strip_elements(root, html.etree.Comment, "script", "style")
            text = html.tostring(root, method="text", encoding=unicode)

            if remove_newlines:
                text = re.sub('\s+', ' ', text)
            self.text = text
        return self.text

    def _get_tokens(self, remove_stopwords=True, min_wordlength=2):
        """ Split text and filter it from rare words and stopwords

        Args:
            text (str): text to split

        Returns: FIXME:
            list of string tokens
            list of `Token` objects with sequence of words preserved
        """
        if not self.tokens:
            text = self._get_text()
            tokens = self.tokenizer.tokenize(text)
            if remove_stopwords:
                pass
            if min_wordlength:
                pass
            self.tokens = tokens
        return self.tokens
        # FIXME: try with stemming all words before lda results
        return [Token(t) for t in self.tokenizer.tokenize(text)]

    @staticmethod
    def _get_tokenizer():
        """ function returns default tonekinzer which is set with help of
        `nltk` library. Default tokenizer is simple regexp tokenizer, which
        cares only about '\w+' tokens.

        Returns:
            tokenizer with method `tokenize`.
        """
        # only alpha words
        return RegexpTokenizer(r'[^a-zA-Z_]+')

    def _text_to_disk(self, folder=None, filename=None, text=None):
        """ Stores text of a link to file in folder `folder` and filename
        `filename`.

        Kwargs:
            folder (string): path of a folder where content of a web page which
                this link points to in a text form
            filename (string): name of a file, where to save text
        """
        assert folder and filename
        # unify interface for saving
        text = text or self.text
        assert text, "There are no text to save !"
        filepath = os.path.join(folder, filename)
        with open(filepath, 'w') as f:
            f.write(text)
            self.log.debug("Written text to file {}".format(filepath))

    def _tokens_to_disk(self, folder=None, filename=None, tokens=None):
        """ Stores tokens, that were extracted from text to
        file in folder `folder` and filename `filename`. Tokens are saved
        as a list and in pickle form.

        Kwargs:
            folder (string): path of a folder where content of a web page which
                this link points to in a tokens form should be stored
            filename (string): name of a file, where to save text
            tokens (list): list of tokens to save
        """
        assert folder and filename and tokens and isinstance(tokens, list)
        tokens = tokens or self.tokens
        assert tokens, "No availale tokens to save to disk!"
        assert isinstance(tokens, list), "Wrong format of tokens!"
        filepath = os.path.join(folder, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(tokens, f)
            self.log.debug("Written tokens to file {}".format(filepath))

    def _tokens_from_disk(self, filepath=None):
        """ Get tokens from disk.

        Kwargs:
            filename (str): filename to read tokens from

        Returns:
            list of tags
        """
        assert filepath and isinstance(filepath, str)
        with open(filepath, 'rb') as f:
            self.tokens = pickle.load(f)
        return self.tokens


class Collection(Logger):

    """ This class manages many separate pages and treats them as distinct
    texts. Theese texts all together form an object which is called Collection.
    """

    pages = None
    texts = None
    dictionary = None
    # sparse vector of features
    corpus = None

    # FIXME: decide how to init instance if we restore texts from file
    def __init__(self, linklist, pages=None, texts=None):
        """ Inits all pages found in linklist.

        Args:
            linklist (list of lists of strings): list of links tuples
            pages (list of Pages objects): list of Pages objects
            texts (list of lists of tokens): of each document
        """
        assert isinstance(linklist, list) and len(linklist) > 1, (
            "Not enough links to proceed!")
        self.linklist = linklist

    def get_tokenized_texts(self):
        """ Extracts tokens from text. Forms and returns list of lists of
        tokens for each documents.
        This method need to downlaod all links in a sequental oreder, so
        it could take a while to wait for it.

        Returns:
            list of lists of tokens from each page
        """
        # init pages for each url
        if not self.texts:
            self.pages = []
            for l in self.linklist:
                self.pages.append(Page(l))
            # form texts
            self.texts = []
            for p in self.pages:
                # catch all excetptions of pages that we cannot parse
                try:
                    tokens = p.get_list_of_tokens()
                except ParsingError:
                    self.log.warn(u"Failed to get tokens from text of url "
                                  "{url}. Skipping it.".format(url=p.url))
                    continue
                self.texts.append(tokens)

            # self.texts = [p.get_list_of_tokens() for p in self.pages]
        return self.texts

    def get_dictionary(self):
        """ Returns gensim dictionary.
        Returns:
            gensim dictionary
        """
        texts = self.get_tokenized_texts()
        if not self.dictionary:
            self.dictionary = corpora.Dictionary(texts)
        return self.dictionary

    def get_corpus(self):
        """TODO: Docstring for get_corpus.
        Returns: TODO

        """
        texts = self.get_tokenized_texts()
        if not self.corpus:
            dictionary = self.get_dictionary()
            self.corpus = [dictionary.doc2bow(text) for text in texts]
        return self.corpus

    def save_corpus_and_dictionary(self, dirname):
        """ save to dirname/corpus.mm and dirname/dictionary.dict """
        corpfilename = os.path.join(dirname, "corpus.mm")
        dictfilename = os.path.join(dirname, "dictionary.dict")
        corpora.MmCorpus.serialize(corpfilename, self.get_corpus())
        self.get(self.get_dictionary()).save(dictfilename)

    def load_corpus_and_dictionary(self, dirname):
        """ Restore saved corpus and dictionary.
        Load from dirname/corpus.mm and dirname/dictionary.dict """
        corpfilename = os.path.join(dirname, "corpus.mm")
        dictfilename = os.path.join(dirname, "dictionary.dict")
        assert os.path.isfile(corpfilename), "Error loading corpus!"
        assert os.path.isfile(dictfilename), "Error loading dictionary!"
        self.dictionary = corpora.Dictionary.load(dictfilename)
        self.corpus = corpora.MmCorpus(corpfilename)


if __name__ == '__main__':
    import logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.DEBUG)
    with open('./links.dump', 'rb') as f:
        links = pickle.load(f)
    col = Collection(links)
    dictionary = col.get_dictionary()
    corpus = col.get_corpus()

"""
with open("./file", "w") as f:
    f.write(lxml.html.tostring(root,
                               method="text",
                               encoding=unicode).encode('utf-8'))

# dump all links to files (for testing purposes)
for l in links:

with open("links.dump", 'rb') as f:
    import pickle
    a = pickle.load(f)
"""

'''
DEPRECATED, but maybe useful
def _get_html(self, url):
    """ Performs request and return raw result in a strint from

    Args:
        url (string): url of a page to get

    Returns:
        raw html string, which contains all the page at once. Resulted html
        page is an unicode object.
        Returns `None` if some erors occured while making request. For
        such an issues look to `WARN` level in logs.
    """
    self.log.debug("Try to get content from page {}".format(url))
    html = None
    try:
        r = requests.get(url)
        ud = UnicodeDammit(r.content, is_html=True)
        html = ud.unicode_markup
    except requests.exceptions.RequestException as e:
        self.log.warn("Unable to get page content of the url: {url}. "
                        "The reason: {exc!r}".format(url=url, exc=e))
        raise ParsingError(e.message)
    return html

@staticmethod
def _get_text_from_html(text, remove_newlines=False):
    """ Cleans all html from string `text`.

    Args:
        text (string): actual string  that contains html code in it
        remove_newlines (bool): wheter perform cleaning of a \n\r
            sequencies or not.

    Returns:
        cleaned text (an unicode object)
    """
    assert isinstance(text, basestring)
    # encode unicode string before parsing it to lxml
    text = text.encode("utf-8")
    root = lxml.html.fromstring(text, encoding="utf-8")
    # ignore alltogether javascript and inline css code
    lxml.etree.strip_elements(root, lxml.etree.Comment, "script", "style")
    text = lxml.html.tostring(root, method="text", encoding=unicode)
    # FIXME: decide if cleaning text is necessary for the next steps
    if remove_newlines:
        text = re.sub('\s+', ' ', text)
    return text

'''
