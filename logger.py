#!/usr/bin/python
import logging

# create logger
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setLevel(logging.DEBUG)
# create formatter
form = u'%(asctime)s - %(name)s - %(levelname)s - %(message)s'
formatter = logging.Formatter(form)
# add formatter to ch
ch.setFormatter(formatter)
# add ch to logger
log.addHandler(ch)


class Logger(object):
    @property
    def log(self):
        name = u'.'.join([__name__, self.__class__.__name__])
        return logging.getLogger(name)
