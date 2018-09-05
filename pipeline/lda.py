import os
from pipeline.similarity_model import SimilarityModel
from gensim.corpora import Dictionary, MmCorpus
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.word2vec import LineSentence
from gensim.similarities import MatrixSimilarity

from pipeline.paths import Paths

class LDABuilder:

    def __init__(self):
        self.paths = Paths()

    def lda_pipeline(self, n_topics=50):
        print('Running LDA pipeline (source: {})'.format(self.paths.subdir))
        trigram_dictionary = self.get_corpus_dict(recalculate=True)
        self.get_trigram_bow_corpus(trigram_dictionary, recalculate=True)
        self.get_model(n_topics, recalculate=True)
        print('Done Done!')


    def get_corpus_dict(self, recalculate=False,  from_scratch=True):

        if not os.path.isfile(self.paths.trigram_dictionary_filepath) or recalculate:

            if not from_scratch:
                raise ValueError('No corpus Dictionary file exists but from_scratch is False')

            print('Building trigram dict...')
            trigram_docs = LineSentence(self.paths.trigram_corpus_filepath)

            # learn the dictionary by iterating over all of the docs
            trigram_dictionary = Dictionary(trigram_docs)

            # filter tokens that are very rare or too common from
            # the dictionary (filter_extremes) and reassign integer ids (compactify)
            trigram_dictionary.filter_extremes(no_below=10, no_above=0.4)
            trigram_dictionary.compactify()

            trigram_dictionary.save(self.paths.trigram_dictionary_filepath)
            print('Done!')
        else:
            print('Loading trigram dict...')
            trigram_dictionary = Dictionary.load(self.paths.trigram_dictionary_filepath)

        return trigram_dictionary


    def get_trigram_bow_corpus(self, trigram_dictionary, recalculate=False, from_scratch=True):

        if not os.path.isfile(self.paths.trigram_bow_filepath) or recalculate:

            if not from_scratch:
                raise ValueError('No BOW corpus file exists but from_scratch is False')

            print('Building bow corpus...')
            trigram_corpus = LineSentence(self.paths.trigram_corpus_filepath)
            # generate bag-of-words representation
            trigram_bow_generator = (trigram_dictionary.doc2bow(doc) for doc in trigram_corpus)
            mm_corpus = MmCorpus.serialize(self.paths.trigram_bow_filepath, trigram_bow_generator)
            print('Done!')
        else:
            print('Loading bow corpus...')
            mm_corpus = MmCorpus(self.paths.trigram_bow_filepath)

        return mm_corpus


    def get_model(self, n_topics=50, n_workers=6, recalculate=False, from_scratch=True):

        filepath = self.paths.get_lda_filepath(n_topics)

        if not os.path.isfile(filepath) or recalculate:

            if not from_scratch:
                raise ValueError('No LDA file exists but from_scratch is False')

            trigram_dictionary = self.get_corpus_dict()
            trigram_bow_corpus = self.get_trigram_bow_corpus(trigram_dictionary)

            print('Building LDA model...')
            lda = LdaMulticore(trigram_bow_corpus,
                               num_topics=n_topics,
                               id2word=trigram_dictionary,
                               workers=n_workers)

            lda.save(filepath)
            print('LDA model (n_topics={}) written to {}'.format(n_topics, filepath))
        else:
            print('Loading LDA model (n_topics={})...'.format(n_topics))
            lda = LdaMulticore.load(filepath)

        return lda


    def get_similarity_index(self, bow_corpus, lda, recalculate=False, from_scratch=True):

        filepath = self.paths.get_lda_index(lda.num_topics)

        if not os.path.isfile(filepath) or recalculate:

            if not from_scratch:
                raise ValueError('No similarity index file exists but from_scratch is False')

            print('Building index...')
            index = MatrixSimilarity(lda[bow_corpus])
            index.save(filepath)
        else:
            print('Loading index...')
            index = MatrixSimilarity.load(filepath)

        return index


    #
    # def choose_topics_subset(self, lda_output, topn=5):
    #     """
    #     Give a subset of topics from LDA output.
    #     """
    #     return [x[0] for x in lda_output[:topn] if x[0] > 0.15]
