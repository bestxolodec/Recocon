
try:
    from selenium import webdriver
    from selenium.common.exceptions import TimeoutException, WebDriverException
    from selenium.common.exceptions import ElementNotVisibleException
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.common.by import By
    # available since 2.4.0
    from selenium.webdriver.support.ui import WebDriverWait
    # available since 2.26.0
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.common.desired_capabilities import DesiredCapabilities

except ImportError as ie:
    print(ie)
    sys.exit('Consider using pip (pip3) to install modules.')

class Searcher(Object):
    """
        Class desinged to perform search on search engines
        and extracts links of results
    """
    next_page_selectors = {
        'google': (By.CSS_SELECTOR, 'a#pnnext'),
        'yandex': (By.CSS_SELECTOR, '.pager__button_kind_next')
    }

    search_results_selectors = {
        'google':  (By.CSS_SELECTOR, selector),
        'yandex':  (By.CSS_SELECTOR, selector)
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
        """"
        Args:
            keyword: string to search
            engine: engine to use
            number_of_links: amount of links we are interesed in


        """"
        assert engine.lower() in search_locations.keys(), (
            "Engine {} is not supported".format(engine))
        self.search_engine = engine
        self.number_of_links = number_of_links
        self.driver = None
        self.links = []
        pass


    def get_links(self):
        """
        Returns:
            List of tuples (l, r), where l - is a link itself (string), and
            r - is a rank number (position of link l on the engine's serp.
        """
        self._get
        self.driver = self._get_webdriver()




    def _get_webdriver(self, driver="chrome"):
        """
        Returns:
            selenium webdriver of specified `driver`
        """
        pass




class Topic :




service_log_path = "{}/chromedriver.log".format("/home/ipaulo/search")
service_args = ['--verbose']
driver = webdriver.Chrome('/home/ipaulo/search/env/bin/chromedriver',
                          service_args=service_args,
                          service_log_path=service_log_path)
# driver = webdriver.Chrome('/home/ipaulo/search/env/bin/chromedriver')
driver.get('http://www.google.com/xhtml');
search_box = driver.find_element_by_name('q')
search_box.send_keys('ChromeDriver')
search_box.submit()
time.sleep(5) # Let the user actually see something!
search_box = driver.find_element_by_name('q')
search_box.send_keys('ChromeDriver')
time.sleep(2)
search_box.submit()
els = driver.find_elements_by_css_selector("ol li h3>a")
for e in els:
    e.text
    e.get_attribute("href")

e = driver.jfind_element_by_css_selector("a#pnnext")



