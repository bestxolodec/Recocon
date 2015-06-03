recogn_texts = filter(POS_filter, lect.text.split())
recogn_texts = map(normal_form, recogn_texts)

doc2bow = id2word.doc2bow(recogn_texts)
# probability of a particular word to stop with
eps =  0.001
topics = sorted((lda[doc2bow]), key=lambda x: x[1], reverse=True)
tagdict = {}
for topicno, prob in topics:
    # this might not word with default installation of gensim (show_ids hack)
    # see https://github.com/piskvorky/gensim/issues/354
    f = filter(lambda x: x[0] > eps,
               lda.show_topic(topicno, topn=None, show_ids=True))
    for word_prob, word_id in f:
        tagdict[word_id] = max(prob*word_prob, tagdict.get(word_id, 0))

taglist = sorted(tagdict, key=tagdict.get, reverse=True)
for t in taglist:

taglist = map(lambda x: id2word[x], taglist)

# show TOP 5 topics using LDA on recognized text
n_best = 5
topn = 30
for num, prob in topics[:n_best]:
    print("Using text from recognition. Probability:{}, topicno: {}, "
          "words: {}\n".format(prob, num, lda.print_topic(num, topn=topn)))

