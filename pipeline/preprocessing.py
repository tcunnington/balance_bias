import spacy
import os
from gensim.models import Phrases
from gensim.models.word2vec import LineSentence
from gensim.parsing.preprocessing import STOPWORDS

from utils import *

from pipeline.utils import Paths

paths = Paths('all_the_news') # punting on this -> TODO make preprocessing a class!

#################################################
#
#     Full Pipeline
#
#################################################

# THIS WILL NEVER BE USED??@QR@#EF
def preprocessing_pipeline(name='all_the_news'):
    nlp = spacy.load('en')
    # starting with a doc per line written "corpus_all.txt"
    write_unigram_sentences()
    bigram_model = get_bigram_model()
    write_bigram_sentences(bigram_model)
    trigram_model = get_trigram_model()
    write_trigram_sentences(trigram_model)
    write_trigram_reviews(nlp, bigram_model, trigram_model)
    # Now we can do LDA and other fancy shit!



#################################################
#
#     Misc  / spaCy
#
#################################################


def lemmatized_sentence_corpus(nlp_model, corpus_filename, batch_size=100, n_threads=6):
    """
    generator- uses spaCy to parse reviews, lemmatize the text, and yield sentences
    """

    for parsed_review in nlp_model.pipe(read_doc_by_line(corpus_filename),
                                        batch_size=batch_size, n_threads=n_threads,
                                        disable=['tagger', 'ner']):

        for sent in parsed_review.sents:
            yield ' '.join([token.lemma_ for token in sent if not punct_space(token)])


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



#################################################
#
#     Unigrams
#
#################################################

def write_unigram_sentences():
    print('load spacy model')
    nlp = spacy.load('en')

    print('segment sentences, write')
    unigram_sentence_itr = add_newline(lemmatized_sentence_corpus(nlp, paths.corpus_filepath))
    batch_write(paths.unigram_sentences_filepath, unigram_sentence_itr)



#################################################
#
#     Bigrams
#
#################################################


def get_bigram_model():
    print('Getting bi-gram model..')
    if not os.path.isfile(paths.bigram_model_filepath):
        print('Loading uni-gram sentences...')
        unigram_sentences = LineSentence(paths.unigram_sentences_filepath)
        print('Building bi-gram model...')
        bigram_model = Phrases(unigram_sentences)  # TODO look into supplying common words to avoid for better phrases
        print('Writing model...')
        bigram_model.save(paths.bigram_model_filepath)
    else:
        bigram_model = Phrases.load(paths.bigram_model_filepath)

    print('Done!')
    return bigram_model


def write_bigram_sentences(bigram_model):
    print('Get unigram sentences..')
    unigram_sentences = LineSentence(paths.unigram_sentences_filepath)
    print('Join bi-gram\'s and write sentences to file...')
    batch_write(paths.bigram_sentences_filepath, (u' '.join(bigram_model[s]) + '\n' for s in unigram_sentences))



#################################################
#
#     Trigrams
#
#################################################


# TODO a possible way to shorten the pipeline??
# trigram = Phrases(bigram[sentence_stream])
# common_terms = ["of", "with", "without", "and", "or", "the", "a"]
# ct_phrases = Phrases(sentence_stream, common_terms=common_terms)
# Phrases(bigram_model(unigram_sentences)[unigram_review]).save(trigram_model_filepath)
# or maybe
# bigram_sentences = bigram_model[unigram_sentences]
# trigram_model = Phrases(bigram_sentences)

def get_trigram_model():
    print('Getting tri-gram model')
    if not os.path.isfile(paths.trigram_model_filepath):
        print('Loading bi-gram sentences...')
        bigram_sentences = LineSentence(paths.bigram_sentences_filepath)
        print('Building tri-gram model...')
        trigram_model = Phrases(bigram_sentences)
        print('Writing model...')
        trigram_model.save(paths.trigram_model_filepath)
    else:
        trigram_model = Phrases.load(paths.trigram_model_filepath)

    print('Done!')
    return trigram_model


def write_trigram_sentences(trigram_model):
    bigram_sentences = LineSentence(paths.bigram_sentences_filepath)

    with codecs.open(paths.trigram_sentences_filepath, 'w', encoding='utf_8') as f:

        for bigram_sentence in bigram_sentences:
            trigram_sentence = u' '.join(trigram_model[bigram_sentence])

            f.write(trigram_sentence + '\n')
        # TODO test: batch_write(trigram_sentences_filepath, (u' '.join(trigram_model[s]) + '\n' for s in bigram_sentences))

# def get_trigram_sentences():
#     if not os.path.isfile(paths.trigram_model_filepath):
#         raise FileExistsError(paths.trigram_model_filepath)
#
#     return LineSentence(trigram_sentences_filepath)

def write_trigram_reviews(nlp, bigram_model, trigram_model):

    with codecs.open(paths.trigram_reviews_filepath, 'w', encoding='utf_8') as f:

        for parsed_review in nlp.pipe(line_review(paths.corpus_filepath),
                                      batch_size=100, n_threads=4, disable=['tagger', 'ner']):
            # lemmatize the text, removing punctuation and whitespace
            unigram_review = [token.lemma_ for token in parsed_review if not punct_space(token)]

            # apply the first-order and second-order phrase models
            bigram_review = bigram_model[unigram_review]
            trigram_review = trigram_model[bigram_review]

            # remove any remaining stopwords
            # TODO can you do this as part of a model?
            trigram_review = [term for term in trigram_review if term not in STOPWORDS]

            # write the transformed review as a line in the new file
            trigram_review = u' '.join(trigram_review)
            f.write(trigram_review + '\n')

