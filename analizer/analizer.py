#!/usr/bin/env python
# encoding: utf-8

import lxml.etree
import lxml.html
import requests
import re
from nltk.stem import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from logger import Logger
import pickle
import os


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


class Link(Logger):

    """
    This class is designed to store all information about particular link,
    grab and clear data from such a link.
    We undoubdetly assume that page downloaded from a link entirely
    fits into memory.
    """

    def __init__(self, link, engine="google", tokenizer=None):
        """
        Args:
            link: typle of (rate, url, description) of a link
            engine:  engine this link comes from
            tokenizer: class that have method `tokenize` and could be called
                the following way: `tokenizer.tokenize(text)`, where text is
                string to be tokenized.
        """
        assert isinstance(link, tuple), "Wrong link format!"
        assert len(link) == 3, "Improper length of link tuple!"
        self.rate, self.url, self.descr = link
        self.tokenizer = tokenizer or self._get_tokenizer()
        # init all initial variables
        self.html = None
        self.text = None
        self.tokens = None

    def get_list_of_tokens(self):
        """ This function retrieves text from url and prepares it for
        further processing: clean from html tags, nonalpha characters,
        stopwords.

        Returns:
            list of tokens in a sequence they have appeared in text
        """
        assert self.url
        self.html = self._get_page_content(self.url)
        self.text = self._clean_from_html(self.html)
        self.tokens = self._tokenize(self.text)

    @staticmethod
    def _get_page_content(url):
        """ Performs request and return raw result in a strint from

        Args:
            url (string): url of a page to get

        Returns:
            raw html string, which contains all the page at once
        """
        raw = requests.get(url)
        return raw.text

    @staticmethod
    def _clean_from_html(text, remove_newlines=False):
        """ Cleans all html from string `text`.

        Args:
            text (string): actual string  that contains html code in it
            remove_newlines (bool): wheter perform cleaning of a \n\r
                sequencies or not.

        Returns:
            cleaned text
        """
        assert isinstance(text, str)
        root = lxml.html.fromstring(text)
        # ignore alltogether javascript and inline css code
        lxml.etree.strip_elements(root, lxml.etree.Comment, "script", "style")
        text = lxml.html.tostring(root, method="text", encoding=unicode)
        # FIXME: decide if cleaning text is necessary for the next steps
        if remove_newlines:
            text = re.sub('\s+', ' ', text)

    @staticmethod
    def _get_tokenizer():
        """ function returns default tonekinzer which is set with help of
        `nltk` library. Default tokenizer is simple regexp tokenizer, which
        cares only about '\w+' tokens.

        Returns:
            tokenizer with method `tokenize`.
        """
        return RegexpTokenizer(r'\w+')

    def _tokenize(self, text):
        """ Split text and filter it from rare words and stopwords

        Args:
            text (str): text to split

        Returns: FIXME:
            list of string tokens
            list of `Token` objects with sequence of words preserved
        """
        return self.tokenizer.tokenize(text)
        # FIXME: try with stemming all words before lda results
        return [Token(t) for t in self.tokenizer.tokenize(text)]

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



if __name__ == '__main__':
    with open('./links.dump', 'rb') as f:
        links = pickle.load(f)
    Links = []

    for l in link:






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
