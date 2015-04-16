from bs4 import UnicodeDammit
import re
import requests
import lxml
import lxml.html
from time import sleep

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


def check(url):
    print "That is url {}".format(url)
    r = requests.get(url)
    ud = UnicodeDammit(r.content, is_html=True)
    content = ud.unicode_markup.encode(ud.original_encoding, "ignore")
    root = lxml.html.fromstring(content)
    lxml.html.etree.strip_elements(root, lxml.etree.Comment,
                                   "script", "style")
    text = lxml.html.tostring(root, method="text", encoding=unicode)
    text = re.sub('\s+', ' ', text)
    print "Text type is {}!".format(type(text))
    print text[:200]


if __name__ == '__main__':
    for url in urls:
        check(url)
