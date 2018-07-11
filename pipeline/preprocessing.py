import spacy
import os
from gensim.models import Phrases
from gensim.models.word2vec import LineSentence
from gensim.parsing.preprocessing import STOPWORDS

from pipeline.utils import *

class Preprocessor():
    def __init__(self, source_name, spacy_model='en'): # OR en_core_web_md OR en_core_web_lg
        self.paths = Paths(source_name)
        print('load spacy model')
        if isinstance(spacy_model,str):
            self.nlp = spacy.load(spacy_model)
        elif isinstance(spacy_model, spacy.lang.en.English):
            self.nlp = spacy_model
        else:
            raise ValueError('Invalid model')

    #################################################
    #
    #     Full Pipeline
    #
    #################################################



    def run_pipeline(self):
        # starting with a doc per line written "corpus_all.txt"
        self.write_unigram_sentences()
        bigram_model = self.get_bigram_model()
        self.write_bigram_sentences(bigram_model)
        trigram_model = self.get_trigram_model()
        self.write_trigram_sentences(trigram_model)
        self.write_trigram_corpus(bigram_model, trigram_model)
        # Now we can do LDA and other fancy shit!



    #################################################
    #
    #     Unigrams
    #
    #################################################

    def write_unigram_sentences(self):
        print('segment sentences, write')
        unigram_sentence_itr = add_newline(lemmatized_sentence_corpus(self.nlp, self.paths.corpus_filepath))
        batch_write(self.paths.unigram_sentences_filepath, unigram_sentence_itr)



    #################################################
    #
    #     Bigrams
    #
    #################################################


    def get_bigram_model(self):
        print('Getting bi-gram model..')
        if not os.path.isfile(self.paths.bigram_model_filepath):
            print('Loading uni-gram sentences...')
            unigram_sentences = LineSentence(self.paths.unigram_sentences_filepath)
            print('Building bi-gram model...')
            bigram_model = Phrases(unigram_sentences)  # TODO look into supplying common words to avoid for better phrases
            print('Writing model...')
            bigram_model.save(self.paths.bigram_model_filepath)
        else:
            bigram_model = Phrases.load(self.paths.bigram_model_filepath)

        print('Done!')
        return bigram_model


    def write_bigram_sentences(self, bigram_model):
        print('Get unigram sentences..')
        unigram_sentences = LineSentence(self.paths.unigram_sentences_filepath)
        print('Join bi-gram\'s and write sentences to file...')
        batch_write(self.paths.bigram_sentences_filepath, (u' '.join(bigram_model[s]) + '\n' for s in unigram_sentences))



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

    def get_trigram_model(self):
        print('Getting tri-gram model')
        if not os.path.isfile(self.paths.trigram_model_filepath):
            print('Loading bi-gram sentences...')
            bigram_sentences = LineSentence(self.paths.bigram_sentences_filepath)
            print('Building tri-gram model...')
            trigram_model = Phrases(bigram_sentences)
            print('Writing model...')
            trigram_model.save(self.paths.trigram_model_filepath)
        else:
            trigram_model = Phrases.load(self.paths.trigram_model_filepath)

        print('Done!')
        return trigram_model


    def write_trigram_sentences(self, trigram_model):
        bigram_sentences = LineSentence(self.paths.bigram_sentences_filepath)
        batch_write(trigram_sentences_filepath, (u' '.join(trigram_model[s]) + '\n' for s in bigram_sentences))

        # with codecs.open(self.paths.trigram_sentences_filepath, 'w', encoding='utf_8') as f:
        #
        #     for bigram_sentence in bigram_sentences:
        #         trigram_sentence = u' '.join(trigram_model[bigram_sentence])
        #
        #         f.write(trigram_sentence + '\n')


    def write_trigram_corpus(self, bigram_model, trigram_model):

        with codecs.open(self.paths.trigram_reviews_filepath, 'w', encoding='utf_8') as f:

            for parsed_doc in self.nlp.pipe(read_doc_by_line(self.paths.corpus_filepath),
                                          batch_size=100, n_threads=4, disable=['tagger', 'ner']):
                # lemmatize the text, removing punctuation and whitespace
                unigram_doc = lemmatize_clean(parsed_doc)

                # apply the first-order and second-order phrase models
                bigram_doc = bigram_model[unigram_doc]
                trigram_doc = trigram_model[bigram_doc]

                # remove any remaining stopwords
                # TODO can you do this as part of a model?
                trigram_doc = [term for term in trigram_doc if term not in STOPWORDS]

                # write the transformed doc as a line in the new file
                trigram_doc = u' '.join(trigram_doc)
                f.write(trigram_doc + '\n')

    def process_doc(self, text):
        """
        Processes a doc completely so it can be used with our trigram-trained LDA model
        """
        bigram_model = self.get_bigram_model()
        trigram_model = self.get_trigram_model()
        # Using spaCy to remove punctuation and lemmatize the text
        nlp = spacy.load('en')
        parsed = nlp(text)
        unigram_doc = lemmatize_clean(parsed)
        # Applying our first-order phrase model to join word pairs
        bigram_doc = bigram_model[unigram_doc]
        # Applying our second-order phrase model to join longer phrases
        trigram_doc = trigram_model[bigram_doc]
        # Removing stopwords
        return [term for term in trigram_doc if term not in STOPWORDS]