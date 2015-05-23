#!/usr/bin/env python
# encoding: utf-8

# import from top level file logging class
from logger import Logger
from time import sleep
from random import randint
# import all selenium staff
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
except ImportError:
        raise ImportError('Please consider installation of'
                          ' a selenium python bindings.')


def sleep_rand(mi=1, ma=3):
    sleep(randint(mi, ma))


class Searcher(Logger):
    """ Class desinged to perform search on search engines
       and extracts links of results
    """
    next_page_selectors = {
        'google': (By.CSS_SELECTOR, 'a#pnnext'),
        'yandex': (By.CSS_SELECTOR, '')  # ???
    }

    search_results_selectors = {
        'google':  (By.CSS_SELECTOR, "ol li h3>a"),
        'yandex':  (By.CSS_SELECTOR, "")  # ???
    }

    input_field_selectors = {
        'google': (By.NAME, 'q'),
        'yandex': (By.NAME, 'text')
    }

    search_locations = {
        'google': 'https://www.google.com/',
        'yandex': 'http://www.yandex.ru/'
    }

    def __init__(self, keyword, engine="google", number_of_links=20):
        """
        Args:
            keyword: string to search
            engine: engine to use
            number_of_links: amount of links we are interesed in

        """
        assert engine.lower() in self.search_locations.keys(), (
            "Engine {} is not supported".format(engine))
        # self.keyword = unicode(keyword.decode('utf-8'))
        self.keyword = keyword
        self.search_engine = engine
        self.number_of_links = number_of_links
        self.driver = None
        self.links = []

    def _get_webdriver(self, driver="chrome"):
        """
        Returns:
            selenium webdriver of specified `driver`
        """
        if driver.lower() == "chrome":
            self.driver = webdriver.Chrome()
        else:
            self.driver = webdriver.Firefox()

    def _start_search(self):
        """ Initialize search with `self.search_engine`
        This function prepare webdriver and actially perform query
        """
        assert self.search_engine
        assert self.driver, "No silenium driver is available!"
        self.driver.get(self.search_locations[self.search_engine])
        typ, sel = self.input_field_selectors[self.search_engine]
        search_box = self.driver.find_element(by=typ, value=sel)
        sleep_rand()
        search_box.send_keys(self.keyword + "\n")
        sleep_rand()
        # for debug purposes:
        """
        service_log_path = "{}/chromedriver.log".format("/home/ipaulo/search")
        service_args = ['--verbose']
        driver = webdriver.Chrome('/home/ipaulo/search/env/bin/chromedriver',
                                   service_args=service_args,
                                   service_log_path=service_log_path)
        """

    def _get_links_from_current_page(self, shift=1):
        assert self.driver, "No silenium driver is available!"
        typ, sel = self.search_results_selectors[self.search_engine]
        els = self.driver.find_elements(by=typ, value=sel)
        self.links.extend([(shift+i+1, e.get_attribute("href"), e.text)
                           for i, e in enumerate(els)])

    def _go_to_next_page(self):
        assert self.driver, "No silenium driver is available!"
        # get type of selector and selector itself
        typ, sel = self.next_page_selectors[self.search_engine]
        nxt = self.driver.find_element(by=typ, value=sel)
        self.driver.get(nxt.get_attribute("href"))

    def get_links(self, number_of_links=None):
        """
        Returns:
            List of tuples (r, l, d), where l - is a link itself (string),
            r - is a rank number (position of link l on the engine's serp,
            d - description of a link
        """
        if number_of_links:
            self.number_of_links = number_of_links
        self._get_webdriver()
        self.log.debug("Webdriver registered: %r" % self.driver)
        self._start_search()
        page = 1
        while True:
            self._get_links_from_current_page(shift=len(self.links))
            # if we have sufficient number of links
            if len(self.links) >= self.number_of_links:
                break
            self.log.debug("Already have {curlinks} links. That is less then"
                           " required {reqlinks}, so moving from {cur} page to "
                           "page {next}.".format(curlinks=len(self.links),
                                                 reqlinks=self.number_of_links,
                                                 cur=page, next=page+1))
            sleep_rand()
            self._go_to_next_page()
            page += 1
        # cut out only those of links that are needed
        links = self.links[:self.number_of_links]
        self.log.debug("All grabbed links: \n {!r}"
                       "".format(links))
        return links

    def exit(self):
        """ Method to clean up and close browser."""
        if self.driver:
            self.driver.close()
            self.driver = None
