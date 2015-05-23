#!/usr/bin/env python
# encoding: utf-8

import os
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.sys.path.insert(0, parentdir)
import argparse
import pickle as pic

from recocon.audio.lecture import Lecture
from recocon.analizer.analizer import Collection
from recocon.searcher.searcher import Searcher

import pymorphy2
morph = pymorphy2.MorphAnalyzer()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Lecture recognizer module. Provides interface only",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # options related to audio processing and recognition
    audio = parser.add_argument_group('audio')
    audio.add_argument(type=str, dest="video_filepath",
                       help="Path to the video file for processing")
    audio.add_argument("-d", "--tmpdir", dest="tmpdir", action="store",
                       default="/tmp/chunks", help="Path to tmp dir, "
                       "where all audio chunks will be stored")
    audio.add_argument("-n", "--noconvert", dest="noconvert",
                       action="store_true", default=False, help="Do not "
                       "extract audio from video, as it was done previously.")
    audio.add_argument("-md", "--model-directory", action="store", type=str,
                       help="Path to the dir containing language model.",
                       dest="model_directory",
                       default="/home/ipaulo/recocon/audio/libs/"
                       "models/zero_ru_cont_8k_v3/")
    audio.add_argument("-lm", dest="lm", action="store",
                       default="ru.lm", help="Path to language "
                       "model file relative to `--model-directory` path."
                       " Required for recognition process.")
    audio.add_argument("-dct", dest="dct", action="store",
                       default="ru.dic", help="Path to language "
                       "dictionary file relative to `--model-directory` path."
                       " Required for recognition process.")
    audio.add_argument("-hmm", dest="hmm", action="store",
                       default="zero_ru.cd_cont_4000", help="Path to directory"
                       " where parameters of Hidden Markov Model are stored."
                       " (relative to `--model-directory` path)"
                       " Required for recognition process.")
    audio.add_argument("-decoder", dest="decoder", action="store",
                       default="pocketsphinx_batch",
                       help="Program name of decoder, that is found in $PATH"
                       ". Defaults to `pocketsphinx_batch`.")
    audio.add_argument("-fr", dest="framerate", action="store",
                       default=8000, type=int,
                       help="Framerate to which convert the video_filepath"
                       " file. Defaults to 8000")
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
    # analizer = parser.add_argument_group('analizer')
    # analizer.add_argument("-k", "--keyword", dest="keyword", required=True,
                          # help="Sentence to search on the web")
    # analizer.add_argument("-e", "--engine", dest="engine", type=str,
                          # help="Engine to use for searching."
                          # " Defaults to google ",  default="google")
    # analizer.add_argument("-l", "--num-links", dest="number_of_links",
                          # help="Number of links to return. "
                          # "Defaults to 20",  default=20, type=int)

    args = parser.parse_args()

    lmfile = os.path.join(args.model_directory, args.lm)
    hmmfile = os.path.join(args.model_directory, args.hmm)
    dctfile = os.path.join(args.model_directory, args.dct)

    lect = Lecture(args.video_filepath, args.tmpdir, decoder=args.decoder,
                   lm=lmfile, dct=dctfile, hmm=hmmfile)

    if not args.noconvert:
        lect.video_to_audio(framerate=args.framerate, bitrate="256k")

    # lect.silence_split()
    # TODO: rename functino name to silence split
    lect.plot_params()

    # pickle results for further purposes
    print(lect.get_full_text())
    lecturefile = os.path.join(args.tmpdir, "./lecture.pickle")
    pic.dump(lect, open(lecturefile, 'wb'))

    # get links from searcher engine's serp
    g = Searcher(args.keyword, engine=args.engine,
                 number_of_links=args.number_of_links)
    links = g.get_links()
    # close browser
    g.exit()

    # prepare collections of texts
    urls = [l[1] for l in links]
    c = Collection(urls)
    texts = c.get_tokenized_texts()

    # FIXME: this functionality should be in separate module
    from gensim import corpora, models
    # load model that was already created in offline
    lda = models.LdaModel.load("./models/lda_on_bow")
    wiki_corpus = corpora.MmCorpus("./models/_bow.mm")

    # create single documents from all grabbed webpages
    all_texts = [word for text in texts for word in text]
    id2word = corpora.Dictionary.load_from_text("./models/_wordids_stripped.txt")
    # corpus is sparse vector of features
    corpus = [id2word.doc2bow(text) for text in texts]
    all_corpus = id2word.doc2bow(all_texts)

    doc2bow = id2word.doc2bow(all_texts)
    # `topics` is a list of tuples: (topicno, probability)
    topics = sorted((lda[doc2bow]), key=lambda x: x[1], reverse=True)
    n_best = 5
    topn = 30
    for num, prob in topics[:n_best]:
        print("Probability:{}, topicno: {}, words: {}\n"
              "".format(prob, num, lda.print_topic(num, topn=topn)))

    # find seconds in what
    # FIXME: awful for's. Thinks about
    #   1. Counting words in dics
    #   2. lemmatizing this search phase
    for ch in lect.chunks:
        for word in ch.text.split():
            lemma = morph.parse(word)[0].normal_form
            for num, prob in topics[:n_best]:
                for topic_w_prob, topic_word in lda.show_topic(num, topn=10):
                    if lemma == topic_word:
                        print("Topic word **{}** matched audio {}-{}"
                              "".format(lemma.upper(), ch.start, ch.end))
                        print("\tTop words of this topic: ```{}```"
                              "".format(list(map(lambda x: x[1],
                                                 lda.show_topic(num)))))

    # make attempt to perform query to LDA model using recognized text from
    # video
    recogn_texts = []
    for word in lect.text.split():
        p = morph.parse(word)[0]
        if p.tag.POS == 'NOUN':
            recogn_texts.append(p.normal_form)
    doc2bow = id2word.doc2bow(recogn_texts)
    topics = sorted((lda[doc2bow]), key=lambda x: x[1], reverse=True)
    # show TOP 5 topics using LDA on recognized text
    n_best = 5
    topn = 30
    for num, prob in topics[:n_best]:
        print("Using text from recognition. Probability:{}, topicno: {}, "
              "words: {}\n".format(prob, num, lda.print_topic(num, topn=topn)))

    # find seconds in what
    # FIXME: awful for's. Thinks about
    #   1. Counting words in dics
    #   2. lemmatizing this search phase
    for ch in lect.chunks:
        for word in ch.text.split():
            lemma = morph.parse(word)[0].normal_form
            for num, prob in topics[:n_best]:
                for topic_w_prob, topic_word in lda.show_topic(num, topn=10):
                    if lemma == topic_word:
                        print("Topic word **{}** matched audio {}-{}"
                              "".format(lemma.upper(), ch.start, ch.end))
                        print("\tTop words of this topic: ```{}```"
                              "".format(list(map(lambda x: x[1],
                                                 lda.show_topic(num)))))
