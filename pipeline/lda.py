import os
import numpy as np
from gensim.corpora import Dictionary, MmCorpus
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.word2vec import LineSentence
from gensim.similarities import MatrixSimilarity

from pipeline.paths import Paths
from pipeline.utils import *

class LDABuilder:

    def __init__(self, source_name):
        self.paths = Paths(source_name)

    def lda_pipeline(self, n_topics=50):
        print('Running LDA pipeline (source: {})'.format(self.paths.subdir))
        trigram_dictionary = self.get_corpus_dict(recalculate=True)
        self.get_trigram_bow_corpus(trigram_dictionary, recalculate=True)
        self.get_lda_model(n_topics, recalculate=True)
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


    def get_lda_model(self, n_topics=50, n_workers=6, recalculate=False, from_scratch=True):

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


    def get_corpus_topics_matrix(self, n_topics=50, recalculate=False, from_scratch=True):

        filepath = self.paths.get_topics_matrix_filepath(n_topics)

        if not os.path.isfile(filepath) or recalculate:

            if not from_scratch:
                raise ValueError('No topics matrix file exists but from_scratch is False')

            lda = self.get_lda_model(n_topics)
            num_topics = lda.num_topics

            trigram_dictionary = self.get_corpus_dict()
            vecs = []

            for parsed_doc in read_doc_by_line(self.paths.trigram_corpus_filepath):
                # doc is already parsed so just needs to remove stopwords and get tokens as list, then to bow
                bow = trigram_dictionary.doc2bow(get_doc_tokens(parsed_doc))

                # topics come in list of (topic#, weight)
                topics = lda[bow]
                topic_vec = self.create_topic_vec(num_topics, topics)
                vecs.append(topic_vec)

                if len(topics) == 0:
                    print('No good article: ' + parsed_doc)

            matrix = np.vstack(vecs)
            row_sums = np.linalg.norm(matrix, axis=1, keepdims=True)
            matrix = matrix / row_sums

            np.save(filepath, matrix)
            print('Topics matrix (n_topics={}) written to {}'.format(n_topics, filepath))
        else:
            print('Loading topics matrix (n_topics={})...'.format(n_topics))
            matrix = np.load(filepath)

        return matrix

    def get_similarity_index(self, bow_corpus, lda, recalculate=False, from_scratch=True):

        filepath = self.paths.get_lda_index(lda.num_topics)

        if not os.path.isfile(filepath) or recalculate:

            if not from_scratch:
                raise ValueError('No similarity index file exists but from_scratch is False')

            index = MatrixSimilarity(lda[bow_corpus])
            index.save(filepath)
        else:
            index = MatrixSimilarity.load(filepath)

        return index

    @staticmethod
    def create_topic_vec(num_topics, topics):

        topic_vec = np.zeros(num_topics)
        if len(topics) == 0:
            print('No topics found, which is strange and should not happen for a real news article.')
            return topic_vec

        [idxs, weights] = list(zip(*topics))
        np.put(topic_vec, idxs, weights)
        return topic_vec


    def cosine_similarity_corpus(self, topics, topics_matrix, n=10):
        topics = topics / np.linalg.norm(topics)
        # corpus topic vectors should already be normalized
        z = 1-np.dot(topics_matrix, topics)
        return np.argpartition(z, -n)[-n:]


    def choose_topics_subset(self, lda_output, topn=5):
        """
        Give a subset of topics from LDA output.
        """
        return [x[0] for x in lda_output[:topn] if x[0] > 0.15]
