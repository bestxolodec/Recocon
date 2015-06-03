WORD_RE = "[^\W\d_]+"
ASCII_RE = "[\x00-\x7F]"

from nltk.corpus import stopwords

rx = re.compile(WORD_RE, re.UNICODE)
filterre = re.compile(ASCII_RE, re.UNICODE)
# extract tokens
tokens = rx.findall(text)
# filter by contains ascii characters
contains_ascii = filter(lambda s: bool(filterre.search(s)), tokens)
tokens = filter(lambda s: not bool(filterre.search(s)), tokens)
# filter by min_wordlength
tokens = filter(lambda s: len(s) > 2, tokens)
contains_less = filter(lambda s: len(s) <= 2, tokens)
# FIXME: learn how to print unicode with logging module
self.log.debug("Excluded from tokens for not satisfying minmial length:"
               "{!r}".format(repr))


stopwords.words('english')

contains_less = filter(lambda s: len(s) <= 3, tokens)
c = Counter(contains_less)
for k in sorted(c.items(), key=operator.itemgetter(1)):
  print k[1], "\t\t", k[0]
