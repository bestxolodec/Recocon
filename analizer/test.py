import pickle
from bs4 import UnicodeDammit
import re
import requests
import lxml
import lxml.html
from lxml.html import etree
import chardet
from lxml import html

THRESHOLD_OF_CHARDETECT = 0.7

urls = [
    "http://mathprofi.ru/zadachi_po_kombinatorike_primery_reshenij.html",
    "http://ru.onlinemschool.com/math/assistance/statistician/",
    "http://mathprofi.ru/zadachi_po_kombinatorike_primery_reshenij.html",
    "http://universarium.org/courses/info/332",
    "http://compsciclub.ru/course/wordscombinatorics",
    "http://ru.onlinemschool.com/math/assistance/statistician/",
    "http://lectoriy.mipt.ru/course/Maths-Combinatorics-AMR-Lects/",
    "http://www.youtube.com/watch?v=SLPrGWQBX0I"
]


# Only this function correctly words
def checkdocument(url):
    print "\n\n"
    print "That is url {}".format(url)
    r = requests.get(url)
    ud = UnicodeDammit(r.content, is_html=True)
    print "\t\t\t\t\t\t", ud.original_encoding == ud.declared_html_encoding
    if not ud.original_encoding == ud.declared_html_encoding:
        print ("Origignal encoding: {} vs declared_html_encoding: {}"
               "".format(ud.original_encoding, ud.declared_html_encoding))
        print "Detected encoding: {!r}". format(chardet.detect(r.content))
    content = ud.unicode_markup.encode(ud.original_encoding, "ignore")
    root = etree.HTML(content,
                      parser=etree.HTMLParser(encoding=ud.original_encoding))
    lxml.html.etree.strip_elements(root, lxml.etree.Comment,
                                   "script", "style")
    text = lxml.html.tostring(root, method="text", encoding="utf-8")
    text = re.sub('\s+', ' ', text)
    print text[:200]


# Only this function correctly words
def check_enc_fixed(url):
    print "\n\n"
    print "That is url {}".format(url)
    r = requests.get(url)
    ud = UnicodeDammit(r.content, is_html=True)
    print "\t\t\t\t\t\t", ud.original_encoding == ud.declared_html_encoding
    if not ud.original_encoding == ud.declared_html_encoding:
        print ("Origignal encoding: {} vs declared_html_encoding: {}"
               "".format(ud.original_encoding, ud.declared_html_encoding))
        print "Detected encoding: {!r}". format(chardet.detect(r.content))

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

    text = re.sub('\s+', ' ', text)
    print text[:200]


if __name__ == '__main__':
    with open('./links.dump', 'rb') as f:
        urls = [link[1] for link in pickle.load(f)]
    for url in urls:
        try:
            check_enc_fixed(url)
        except Exception as e:
            print str(e)
