import os

import spacy
from gensim.corpora import Dictionary, MmCorpus
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.word2vec import LineSentence

from pipeline.utils import *

paths = Paths('all_the_news')


def lda_pipeline(n_topics=50):
    print('Getting trigram dict...')
    trigram_dictionary = get_corpus_dict()
    print('Getting bow corpus...')
    get_trigram_bow_corpus(trigram_dictionary)
    print("Building LDA model...")
    get_lda_model(n_topics) # will just save for use later
    print('Done!')


def get_corpus_dict(recalculate=False):

    if not os.path.isfile(paths.trigram_dictionary_filepath) or recalculate:
        trigram_reviews = LineSentence(paths.trigram_reviews_filepath)

        # learn the dictionary by iterating over all of the reviews
        trigram_dictionary = Dictionary(trigram_reviews)

        # filter tokens that are very rare or too common from
        # the dictionary (filter_extremes) and reassign integer ids (compactify)
        trigram_dictionary.filter_extremes(no_below=10, no_above=0.4)
        trigram_dictionary.compactify()

        trigram_dictionary.save(paths.trigram_dictionary_filepath)
    else:
        # load the finished dictionary from disk
        trigram_dictionary = Dictionary.load(paths.trigram_dictionary_filepath)

    return trigram_dictionary


def get_trigram_bow_corpus(trigram_dictionary, recalculate=False):

    if not os.path.isfile(paths.trigram_bow_filepath) or recalculate:
        trigram_corpus = LineSentence(paths.trigram_reviews_filepath)
        # generate bag-of-words representation
        trigram_bow_generator = (trigram_dictionary.doc2bow(doc) for doc in trigram_corpus)
        MmCorpus.serialize(paths.trigram_bow_filepath, trigram_bow_generator)

    # load the finished bag-of-words corpus from disk
    return MmCorpus(paths.trigram_bow_filepath)


def get_lda_model(n_topics=50, n_workers=6, recalculate=False, from_scratch=True):

    filepath = paths.get_lda_filepath(n_topics)

    if not os.path.isfile(filepath) or recalculate:

        if not from_scratch:
            raise ValueError('No LDA file exists but from_scratch is False')

        print('Building LDA model...')
        trigram_dictionary = get_corpus_dict()
        trigram_bow_corpus = get_trigram_bow_corpus(trigram_dictionary)

        lda = LdaMulticore(trigram_bow_corpus,
                           num_topics=n_topics,
                           id2word=trigram_dictionary,
                           workers=n_workers)

        lda.save(filepath)
        print('LDA model written to ' + filepath)
    else:
        print('Loading LDA model from ' + filepath)
        lda = LdaMulticore.load(filepath)

    return lda


def trigram_doc_to_bow(trigram_doc):
    # Creating a bag-of-words representation
    trigram_dictionary = get_corpus_dict()
    return trigram_dictionary.doc2bow(trigram_doc)


def choose_topics_subset(lda_output, topn=1):
    """
    Give a subset of topics from LDA output.
    """
    return [x[0] for x in lda_output[:topn]]