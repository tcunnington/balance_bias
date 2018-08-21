import os


class Paths:
    def __init__(self):

        # preprocessing
        self.base = 'pipeline'
        self.data_directory              = os.path.join(self.base, 'data')
        self.intermediate_directory      = os.path.join(self.base, 'intermediate')
        self.output_directory            = os.path.join(self.base, 'output')
        self.unigram_sentences_filepath  = os.path.join(self.intermediate_directory, 'unigram_sentences_all.txt')
        self.corpus_filepath             = os.path.join(self.intermediate_directory, 'corpus_all.txt')
        self.bigram_model_filepath       = os.path.join(self.intermediate_directory, 'bigram_model_all') # bigram.model
        self.bigram_sentences_filepath   = os.path.join(self.intermediate_directory, 'bigram_sentences_all.txt')
        self.trigram_model_filepath      = os.path.join(self.intermediate_directory, 'trigram_model_all') # trigram.model
        self.trigram_sentences_filepath  = os.path.join(self.intermediate_directory, 'trigram_sentences_all.txt')
        self.corpus_meta_data            = os.path.join(self.intermediate_directory, 'corpus_meta.csv')

        # models
        self.models                      = os.path.join(self.base, 'models')
        self.trigram_corpus_filepath     = os.path.join(self.intermediate_directory, 'trigram_transformed_corpus_all.txt')
        self.trigram_dictionary_filepath = os.path.join(self.intermediate_directory, 'trigram_dict_all.dict')
        self.trigram_bow_filepath        = os.path.join(self.intermediate_directory, 'trigram_bow_corpus_all.mm')
        # self.lda_model_filepath          = os.path.join(self.intermediate_directory, 'lda.model')
        # self.word2vec_filepath           = os.path.join(self.intermediate_directory, 'word2vec_model_all')

    def data_file(self, file):
        return os.path.join(self.data_directory, file)

    def get_lda_filepath(self, n_topics):
        return os.path.join(self.models, 'lda.n' + str(int(n_topics)) + '.model')

    def get_ldaviz_filepath(self, n_topics):
        return os.path.join(self.models, 'ldaviz.n' + str(int(n_topics)) + '.model')

    # def get_topics_matrix_filepath(self, n_topics):
    #     return os.path.join(self.intermediate_directory, 'topics.n' + str(int(n_topics)) + '.npy')

    def get_lda_index(self, n_topics):
        return os.path.join(self.models, 'similarity.n' + str(int(n_topics)))

    def output_file(self, file):
        return os.path.join(self.output_directory, file)

    def ldaviz_json(self, n_topics):
        return self.output_file('ldaviz.n{}.json'.format(str(int(n_topics))))

    def get_lsa_filepath(self, n_topics):
        return os.path.join(self.models, 'lsa.n' + str(int(n_topics)) + '.model')

    def get_lsa_index(self, n_topics): # TODO merge to create general index path function
        return os.path.join(self.models, 'similarity_lsa.n' + str(int(n_topics)))

    def get_doc2vec(self, n_dims):
        return os.path.join(self.models, 'doc2vec.n' + str(int(n_dims)) + '.model')