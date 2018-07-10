import os

from gensim.corpora import Dictionary, MmCorpus
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.word2vec import LineSentence

import pyLDAvis
import pyLDAvis.gensim
import pickle

from pipeline.utils import Paths

paths = Paths('all_the_news')


def lda_pipeline():
    print('Getting trigram dict...')
    trigram_dictionary = get_corpus_dict()
    print('Getting bow corpus...')
    bow = get_trigram_bow_corpus(trigram_dictionary)
    print("Building LDA model...")
    lda = get_lda_model() # will just save for use later
    print('Building LDA viz model...')
    get_ldaviz_model(lda, bow, trigram_dictionary) # will just save for use later
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
        trigram_bow_generator = (trigram_dictionary.doc2bow(review) for review in trigram_corpus)
        MmCorpus.serialize(paths.trigram_bow_filepath, trigram_bow_generator)

    # load the finished bag-of-words corpus from disk
    return MmCorpus(paths.trigram_bow_filepath)


def get_lda_model(recalculate=False, n_topics=50, n_workers=6):

    if not os.path.isfile(paths.lda_model_filepath) or recalculate:
        trigram_dictionary = get_corpus_dict()
        trigram_bow_corpus = get_trigram_bow_corpus(trigram_dictionary)

        lda = LdaMulticore(trigram_bow_corpus,
                           num_topics=n_topics,
                           id2word=trigram_dictionary,
                           workers=n_workers)

        lda.save(paths.lda_model_filepath)
    else:
        lda = LdaMulticore.load(paths.lda_model_filepath)

    return lda


def get_ldaviz_model(lda_model, trigram_bow_corpus, trigram_dictionary, recalculate=False):

    if not os.path.isfile(paths.ldavis_data_filepath) or recalculate:
        LDAvis_prepared = pyLDAvis.gensim.prepare(lda_model, trigram_bow_corpus,
                                                  trigram_dictionary)

        with open(paths.ldavis_data_filepath, 'wb') as f:
            pickle.dump(LDAvis_prepared, f)
    else:
        with open(paths.ldavis_data_filepath) as f:
            LDAvis_prepared = pickle.load(f)

    return LDAvis_prepared
