import logging
import requests
import chardet
from bs4 import UnicodeDammit
import lxml
from lxml.cssselect import CSSSelector
from time import sleep


# global compiled selectors
selectors = [CSSSelector(s) for s in ["h1", "h2", "h3", "h4", "h5"]]


def getheaders(url):
    logging.info("Trying to fetch url: {}".format(url))
    r = requests.get(url, timeout=1)
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
        if enc == det_enc and det_conf < 0.7:
            enc = declared_enc
    content = r.content.decode(enc, "ignore").encode(enc)
    htmlparser = lxml.html.etree.HTMLParser(encoding=enc)
    searchtree = lxml.html.fromstring(content, parser=htmlparser)

    headers = []
    for sel in selectors:
        texts = []
        for h in sel(searchtree):
            texts.append(h.text)
        headers.append(texts)
    return headers


def iter_headers(urls):
    for u in urls:
        try:
            print(getheaders(u))
            print()
            sleep(1)
        except Exception as e:
            logging.error(str(e))
