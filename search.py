#!/usr/bin/env python
# encoding: utf-8

from __future__ import division
import os
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.sys.path.insert(0, parentdir)
from searcher.searcher import Searcher
from logger import Logger

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Lecture recognizer module.")




    # searcher related options
    searcher = parser.add_argument_group('searcher')
    searcher.add_argument("-k", "--keyword", dest="keyword", required=True,
                          help="Sentence to search on the web")
    searcher.add_argument("-e", "--engine", dest="engine", type=str,
                          help="Engine to use for searching."
                          " Defaults to google ",  default="google")
    searcher.add_argument("-l", "--num-links", dest="number_of_links",
                          help="Number of links to return. "
                          "Defaults to 20",  default=20, type=int)
    # analizer related options
    analizer = parser.add_argument_group('analizer')
    analizer.add_argument("-k", "--keyword", dest="keyword", required=True,
                          help="Sentence to search on the web")
    analizer.add_argument("-e", "--engine", dest="engine", type=str,
                          help="Engine to use for searching."
                          " Defaults to google ",  default="google")
    analizer.add_argument("-l", "--num-links", dest="number_of_links",
                          help="Number of links to return. "
                          "Defaults to 20",  default=20, type=int)







    args = parser.parse_args()
    g = Searcher(args.keyword, engine=args.engine,
                 number_of_links=args.number_of_links)
    print args
    links = g.get_links()



