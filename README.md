# recocon
Python proof of concept of ability to segment video into topics.
Project is devided in three parts:
  * Audio - audio recognition using CMU Shpinx
  * Searcher - web search crawler using Silenium
  * Analizer - ?


## Audio
Dependencies:
  * numpy
  * cmu chpinx
  * FIXME

## Searcher
Performs searching of a keyword in google and collect first `N` links.

Steps to get things ready:
  * install selenium either with `pip` or using `apt-get install python-selenium` command
  * install chrome driver from [here](https://sites.google.com/a/chromium.org/chromedriver/downloads) to use chrome and place it somewhere in the `$PATH`

## Analizer

Performs:
  * links crawling
  * cleaning them from html
  * clean them from ascii symbols
  * tokenize and lowercase resulted text


### Installation

Steps to get things ready:
  * python-virtualenv
  * libxslt-dev
  * libxml2-dev
  * libxslt1-dev
  * python3-dev
  * python-dev
  * libz-dev (to fix `/usr/bin/ld: cannot find -lz` problem)


Steps to get things ready:
  * install development packages
    ```
    apt-get install python-dev libxml2-dev libxslt1-dev  python-dev python3-dev libxml2-dev  zlib1g-dev lib32z1-dev chromium-browser   python-ipdb  build-essential cmake libblas-dev liblapack-dev gfortran python-openssl libffi-dev libssl-dev libxml2 gcc automake autoconf libtool bison swig gcc automake autoconf libtool bison swig atop sox lrzip libav-tools  gfortrank
    ```
  * install python packages inside a vritaulenv
  ```
  beautifulsoup4 (4.3.2)
  boto (2.38.0)
  bz2file (0.98)
  chardet (2.3.0)
  DAWG-Python (0.7.2)
  docopt (0.6.2)
  gensim (0.11.1-1)
  ipython (3.1.0)
  lxml (3.4.4)
  matplotlib (1.4.3)
  nltk (3.0.2)
  nose (1.3.6)
  numpy (1.9.2)
  pip (1.5.4)
  pymorphy2 (0.8)
  pymorphy2-dicts (2.4.393442.3710985)
  pyparsing (2.0.3)
  python-dateutil (2.4.2)
  pytz (2015.2)
  requests (2.7.0)
  scipy (0.15.1)
  selenium (2.45.0)
  setuptools (2.2)
  six (1.9.0)
  smart-open (1.2.1)
  ```


