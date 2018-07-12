import os


class Paths:
    def __init__(self, subdir):
        self.subdir = subdir

        # preprocessing
        self.base = 'pipeline'
        self.data_directory              = os.path.join(self.base, 'data', self.subdir)
        self.intermediate_directory      = os.path.join(self.base, 'intermediate', self.subdir)
        self.output_directory            = os.path.join(self.base, 'output', self.subdir)
        self.unigram_sentences_filepath  = os.path.join(self.intermediate_directory, 'unigram_sentences_all.txt')
        self.corpus_filepath             = os.path.join(self.intermediate_directory, 'corpus_all.txt')
        self.bigram_model_filepath       = os.path.join(self.intermediate_directory, 'bigram_model_all') # bigram.model
        self.bigram_sentences_filepath   = os.path.join(self.intermediate_directory, 'bigram_sentences_all.txt')
        self.trigram_model_filepath      = os.path.join(self.intermediate_directory, 'trigram_model_all') # trigram.model
        self.trigram_sentences_filepath  = os.path.join(self.intermediate_directory, 'trigram_sentences_all.txt')
        # self.trigram_reviews_filepath    = os.path.join(self.intermediate_directory, 'trigram_transformed_reviews_all.txt')
        self.trigram_corpus_filepath = os.path.join(self.intermediate_directory, 'trigram_transformed_corpus_all.txt')

        # models
        self.trigram_dictionary_filepath = os.path.join(self.intermediate_directory, 'trigram_dict_all.dict')#'trigram.dict')
        self.trigram_bow_filepath        = os.path.join(self.intermediate_directory, 'trigram_bow_corpus_all.mm')
        self.lda_model_filepath          = os.path.join(self.intermediate_directory, 'lda.model')
        # self.topic_names_filepath = os.path.join(intermediate_directory, 'topic_names.pkl')
        self.ldavis_data_filepath        = os.path.join(self.intermediate_directory, 'ldavis_prepared.model')
        self.word2vec_filepath           = os.path.join(self.intermediate_directory, 'word2vec_model_all')

    def data_file(self, file):
        return os.path.join(self.data_directory, file)

    def get_lda_filepath(self, n_topics):
        return os.path.join(self.intermediate_directory, 'lda.n' + str(int(n_topics)) + '.model')

    def output_file(self, file):
        return os.path.join(self.output_directory, file)