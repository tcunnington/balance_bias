import codecs
import os
import re

class Paths:
    def __init__(self, subdir):
        self.subdir = subdir

        # preprocessing
        base = 'pipeline'
        self.data_directory              = os.path.join(base, 'data', self.subdir)
        self.intermediate_directory      = os.path.join(base, 'intermediate', self.subdir)
        self.unigram_sentences_filepath  = os.path.join(self.intermediate_directory, 'unigram_sentences_all.txt')
        self.corpus_filepath             = os.path.join(self.intermediate_directory, 'corpus_all.txt')
        self.bigram_model_filepath       = os.path.join(self.intermediate_directory, 'bigram_model_all') # bigram.model
        self.bigram_sentences_filepath   = os.path.join(self.intermediate_directory, 'bigram_sentences_all.txt')
        self.trigram_model_filepath      = os.path.join(self.intermediate_directory, 'trigram_model_all') # trigram.model
        self.trigram_sentences_filepath  = os.path.join(self.intermediate_directory, 'trigram_sentences_all.txt')
        self.trigram_reviews_filepath    = os.path.join(self.intermediate_directory, 'trigram_transformed_reviews_all.txt')
        # TODO use this instead of the above
        self.trigram_corpus_filepath = os.path.join(self.intermediate_directory, 'trigram_transformed_corpus_all.txt')

        # models
        self.trigram_dictionary_filepath = os.path.join(self.intermediate_directory, 'trigram_dict_all.dict')#'trigram.dict')
        self.trigram_bow_filepath        = os.path.join(self.intermediate_directory, 'trigram_bow_corpus_all.mm')
        self.lda_model_filepath          = os.path.join(self.intermediate_directory, 'lda.model')
        # self.topic_names_filepath = os.path.join(intermediate_directory, 'topic_names.pkl')
        self.ldavis_data_filepath        = os.path.join(self.intermediate_directory, 'ldavis_prepared.model')
        self.word2vec_filepath           = os.path.join(self.intermediate_directory, 'word2vec_model_all')

    def data_file(self, file):
        return os.path.join('..', 'data', self.subdir, file)

    def get_lda_filepath(self, n_topics):
        return os.path.join(self.intermediate_directory, 'lda.n' + str(int(n_topics)) + '.model')

def batch_write(wfilename, iterator, batch_size=100):
    """
    Write files by the batch from an iterator. Created under the (possibly mistaken) belief that it's faster than
    asking for many small writes
    """
    count = 0
    with codecs.open(wfilename, 'w', encoding='utf_8') as f:
        batch = []
        for item in iterator:
            batch.append(item)
            count += 1

            if len(batch) == batch_size:
                f.write(''.join(batch))
                batch = []

        # final write for partial patch
        f.write(''.join(batch))
        print(u'''Wrote {:,} items to the new txt file '{}'.'''.format(count, wfilename))

def add_newline(itr):
    """
    generator wrapper to add newlines to items of an iterator
    """
    for item in itr:
        yield item + '\n'

def prep_whitespace(text):
    return re.sub(r'\s+', ' ', text)


#################################################
#
#     Misc  / spaCy
#
#################################################


def is_punct_space(token):
    """
    helper function to find tokens that are pure punctuation or whitespace
    """
    return token.is_punct or token.is_space

def read_doc_by_line(filename):
    """
    generator to read in docs from the file and un-escape the original line breaks in the text
    """

    with codecs.open(filename, encoding='utf_8') as f:
        for doc in f:
            yield doc.replace('\\n', '\n')

def lemmatize_clean(spacy_itr):
    """
    generator to act on a spaCy object that allows iterating over tokens
    """
    return (token.lemma_ for token in spacy_itr if not is_punct_space(token))