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
    #                       help="Sentence to search on the web")
    # analizer.add_argument("-e", "--engine", dest="engine", type=str,
    #                       help="Engine to use for searching."
    #                       " Defaults to google ",  default="google")
    # analizer.add_argument("-l", "--num-links", dest="number_of_links",
    #                       help="Number of links to return. "
    #                       "Defaults to 20",  default=20, type=int)

    args = parser.parse_args()

    lmfile = os.path.join(args.model_directory, args.lm)
    hmmfile = os.path.join(args.model_directory, args.hmm)
    dctfile = os.path.join(args.model_directory, args.dct)

    lect = Lecture(args.video_filepath, args.tmpdir, decoder=args.decoder,
                   lm=lmfile, dct=dctfile, hmm=hmmfile)

    if not args.noconvert:
        lect.video_to_audio(framerate=args.framerate, bitrate="256k")

    # lect.silence_split()
    # TODO: rename function name to silence split
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

    # FIXME: the folowing functionality should be in separate module
    from gensim import corpora, models
    # load model that was already created in offline
    lda = models.LdaModel.load("./models/lda_on_bow")
    tfidf_wiki = models.TfidfModel.load("./models/origin.tfidf_model")
    corpus_wiki = corpora.MmCorpus("./models/_bow.mm")
    id2word = corpora.Dictionary.load_from_text("./models/"
                                                "_wordids_stripped.txt")

    # create single documents from all scraped webpages
    texts_flat = [word for text in texts for word in text]
    corpus_scraped_flat = id2word.doc2bow(texts_flat)
    corpus_scraped = [id2word.doc2bow(text) for text in texts]
    doc2bow = id2word.doc2bow(texts_flat)


    # `topics` is a list of tuples: (topicno, probability)
    # lda[doc2bow] infers only those of topics which have probability
    # more then 0.01
    topics = sorted((lda[doc2bow]), key=lambda x: x[1], reverse=True)
    n_best = 5
    topn = 30
    for num, prob in topics[:n_best]:
        print("Probability:{}, topicno: {}, words: {}\n"
              "".format(prob, num, lda.print_topic(num, topn=topn)))

    def normal_form(word):
        return morph.parse(word)[0].normal_form

    def POS_filter(word, pos_list=['NOUN']):
        p = morph.parse(word)[0]
        if p.tag.POS in pos_list:
            return True
        else:
            return False

    # # TRY TO FIGURE OUT TAGS FOR TEXT BASED ON SCRAPED RESULTS
    # # find seconds in what
    # # FIXME: awful for's. Thinks about
    # #   1. Counting words in dics
    # #   2. lemmatizing this search phase
    # for ch in lect.chunks:
    #     for word in ch.text.split():
    #         lemma = normal_form(word)
    #         for num, prob in topics[:n_best]:
    #             for topic_w_prob, topic_word in lda.show_topic(num, topn=10):
    #                 if lemma == topic_word:
    #                     print("Topic word **{}** matched audio {}-{}"
    #                           "".format(lemma.upper(), ch.start, ch.end))
    #                     print("\tTop words of this topic: ```{}```"
    #                           "".format(list(map(lambda x: x[1],
    #                                              lda.show_topic(num)))))

    # # make attempt to perform query to LDA model using recognized text from
    # # video
    # recogn_texts = filter(POS_filter, lect.text.split())
    # recogn_texts = map(normal_form, recogn_texts)

    # doc2bow = id2word.doc2bow(recogn_texts)
    # topics = sorted((lda[doc2bow]), key=lambda x: x[1], reverse=True)
    # # show TOP 5 topics using LDA on recognized text
    # n_best = 5
    # topn = 30
    # for num, prob in topics[:n_best]:
    #     print("Using text from recognition. Probability:{}, topicno: {}, "
    #           "words: {}\n".format(prob, num, lda.print_topic(num, topn=topn)))

    # # find seconds in what
    # # FIXME: awful for's. Think about
    # #   1. Counting words in dics
    # #   2. lemmatizing this search phase
    # for ch in lect.chunks:
    #     for word in ch.text.split():
    #         lemma = morph.parse(word)[0].normal_form
    #         for num, prob in topics[:n_best]:
    #             for topic_w_prob, topic_word in lda.show_topic(num, topn=10):
    #                 if lemma == topic_word:
    #                     print("Topic word **{}** matched audio {}-{}"
    #                           "".format(lemma.upper(), ch.start, ch.end))
    #                     print("\tTop words of this topic: ```{}```"
    #                           "".format(list(map(lambda x: x[1],
    #                                              lda.show_topic(num)))))

    # FIXME: decide what value is appropriate here
    top_n_nearest = len(corpus_scraped) // 4
    # doc2bow is representation of recognized text
    # basis is a list of mappings (topicno, probability)
    basis = lda.__getitem__(doc2bow, eps=0)
    basis = dict(basis)

    # fill in dict of all information about resources we processed
    resources = dict()
    # numerical order of links (resources)
    # rank link description
    for key, (r, l, d) in enumerate(links):
        resources[key] = {'link': l, 'rank': r, 'description': d,
                          'tokens': texts[key]}

    def abs_similarity(basis, doc):
        abs_sim = 0
        if len(doc) == 0:
            # 2 stands for maximal abs similarity between two distributions
            return 2
        for num, prob in doc.items():
            basis_prob = basis[num]
            # print("abs_sim += abs({} - {}) = {}".format(basis_prob, prob,
            #                                        basis_prob - prob))
            abs_sim += abs(basis_prob - prob)
        return abs_sim

    for key, c in enumerate(corpus_scraped):
        c = dict(lda[c])
        resources[key]['similarity'] = abs_similarity(basis, c)

    top_nearest = sorted(resources, key=lambda x:
                         resources[x]['similarity'])[:top_n_nearest]

    recogn_texts = filter(POS_filter, lect.text.split())
    recogn_texts = map(normal_form, recogn_texts)
    doc2bow = id2word.doc2bow(recogn_texts)

    # build tfidf model for all scraped texts
    tfidf_scraped = models.TfidfModel(corpus_scraped, id2word=id2word)
    corpus_scraped_nearest = [corpus_scraped[t] for t in top_nearest]
    tfidf_scraped_nearest = models.TfidfModel(corpus_scraped_nearest,
                                              id2word=id2word)

    res_tfidf_wiki = tfidf_wiki[doc2bow]
    res_tfidf_scraped = tfidf_scraped[doc2bow]
    res_tfidf_scraped_nearest = tfidf_scraped_nearest[doc2bow]

    for num, prob in sorted(res_tfidf_wiki, key=lambda x: x[1], reverse=True)[:20]:
            print(prob, id2word[num])

    for num, prob in sorted(res_tfidf_scraped, key=lambda x: x[1], reverse=True)[:20]:
            print(prob, id2word[num])

    for num, prob in sorted(res_tfidf_scraped_nearest, key=lambda x: x[1], reverse=True)[:20]:
            print(prob, id2word[num])
