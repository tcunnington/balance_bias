import os

from gensim.models.lsimodel import LsiModel
from gensim.models.word2vec import LineSentence
from gensim.similarities import MatrixSimilarity
from pipeline.paths import Paths



class LSABuilder:

    def __init__(self, source_name, lda_builder):
        self.paths = Paths()
        # piggyback off lda builder for now...
        self.lda_builder = lda_builder


    def pipeline(self, n_topics=50):
        print('Running LSA pipeline (source: {})'.format(self.paths.subdir))
        self.get_lsa_model(n_topics, recalculate=True)
        print('Done Done!')

    def get_lsa_model(self, n_topics=50, recalculate=False, from_scratch=True):

        filepath = self.paths.get_lsa_filepath(n_topics)

        if not os.path.isfile(filepath) or recalculate:

            if not from_scratch:
                raise ValueError('No LSA file exists but from_scratch is False')

            trigram_dictionary = self.lda_builder.get_corpus_dict()
            trigram_bow_corpus = self.lda_builder.get_trigram_bow_corpus(trigram_dictionary)

            print('Building LSA model...')
            lsi = LsiModel(trigram_bow_corpus, id2word=trigram_dictionary, num_topics=n_topics)

            lsi.save(filepath)
            print('LSA model (n_topics={}) written to {}'.format(n_topics, filepath))
        else:
            print('Loading LSA model (n_topics={})...'.format(n_topics))
            lsi = LsiModel.load(filepath)

        return lsi

    # TODO move to general function
    def get_similarity_index(self, bow_corpus, lsa: LsiModel, recalculate=False, from_scratch=True):

        filepath = self.paths.get_lsa_index(lsa.num_topics)

        if not os.path.isfile(filepath) or recalculate:

            if not from_scratch:
                raise ValueError('No similarity index file exists but from_scratch is False')

            print('Building index...')
            index = MatrixSimilarity(lsa[bow_corpus])
            index.save(filepath)
        else:
            print('Loading index...')
            index = MatrixSimilarity.load(filepath)

        return index