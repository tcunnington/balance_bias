import spacy
import os
from gensim.models.phrases import Phrases, Phraser
from gensim.models.word2vec import LineSentence
from gensim.parsing.preprocessing import STOPWORDS

from pipeline.paths import Paths
from pipeline.utils import *

class Preprocessor():
    """
    Handles document parsing, lemmatization, and phrase modelling
    """

    def __init__(self, source_name, spacy_model='en', preload_models=False): # OR en_core_web_md OR en_core_web_lg
        self.paths = Paths(source_name)
        print('load spacy model')
        if isinstance(spacy_model,str):
            self.nlp = spacy.load(spacy_model)
        elif isinstance(spacy_model, spacy.lang.en.English):
            self.nlp = spacy_model
        else:
            raise ValueError('Invalid model')

        if preload_models:
            try:
                self.bigram_model = self.get_bigram_model(from_scratch=False)
                self.trigram_model = self.get_trigram_model(from_scratch=False)
            except ValueError:
                raise ValueError('Cannot preload models if they haven\'t already been generated')
        else:
            self.bigram_model = None
            self.trigram_model = None

    #################################################
    #
    #     Full Pipeline
    #
    #################################################


    def run_pipeline(self):
        # starting with a doc per line written "corpus_all.txt"
        self.write_unigram_sentences()
        bigram_model = self.get_bigram_model(recalculate=True)
        self.write_bigram_sentences(bigram_model)
        trigram_model = self.get_trigram_model(recalculate=True)
        self.write_trigram_sentences(trigram_model)
        self.write_trigram_corpus(bigram_model, trigram_model)



    #################################################
    #
    #     Unigrams
    #
    #################################################

    def write_unigram_sentences(self):
        print('segment sentences, write')
        unigram_sentence_itr = add_newline(self.lemmatized_sentence_corpus())
        batch_write(self.paths.unigram_sentences_filepath, unigram_sentence_itr)



    #################################################
    #
    #     Bigrams
    #
    #################################################


    def get_bigram_model(self, recalculate=False, from_scratch=True):

        if not os.path.isfile(self.paths.bigram_model_filepath) or recalculate:

            if not from_scratch:
                raise ValueError('No bigram model file exists but from_scratch is False')

            print('Building bi-gram model...')
            unigram_sentences = LineSentence(self.paths.unigram_sentences_filepath)
            bigram_model = Phrases(unigram_sentences)  # TODO look into supplying common words to avoid for better phrases
            bigram_model = Phraser(bigram_model)
            print('Writing model...')
            bigram_model.save(self.paths.bigram_model_filepath)
        else:
            print('Loading bi-gram model...')
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

    def get_trigram_model(self, recalculate=False, from_scratch=True):

        if not os.path.isfile(self.paths.trigram_model_filepath) or recalculate:

            if not from_scratch:
                raise ValueError('No trigram model file exists but from_scratch is False')

            print('Building tri-gram model...')
            bigram_sentences = LineSentence(self.paths.bigram_sentences_filepath)
            trigram_model = Phrases(bigram_sentences)
            trigram_model = Phraser(trigram_model)
            print('Writing model...')
            trigram_model.save(self.paths.trigram_model_filepath)
        else:
            print('Loading tri-gram model...')
            trigram_model = Phrases.load(self.paths.trigram_model_filepath)

        print('Done!')
        return trigram_model


    def write_trigram_sentences(self, trigram_model):
        bigram_sentences = LineSentence(self.paths.bigram_sentences_filepath)
        batch_write(self.paths.trigram_sentences_filepath, (u' '.join(trigram_model[s]) + '\n' for s in bigram_sentences))

        # with codecs.open(self.paths.trigram_sentences_filepath, 'w', encoding='utf_8') as f:
        #
        #     for bigram_sentence in bigram_sentences:
        #         trigram_sentence = u' '.join(trigram_model[bigram_sentence])
        #
        #         f.write(trigram_sentence + '\n')


    def write_trigram_corpus(self, bigram_model, trigram_model, batch_size=100, n_threads=6):

        with codecs.open(self.paths.trigram_corpus_filepath, 'w', encoding='utf_8') as f:

            for parsed_doc in self.parse_corpus_by_line():
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
        bigram_model = self.bigram_model or self.get_bigram_model()
        trigram_model = self.trigram_model or self.get_trigram_model()
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

    #################################################
    #
    #     Misc
    #
    #################################################

    def lemmatized_sentence_corpus(self, ):
        """
        generator- uses spaCy to parse docs, lemmatize the text, and yield sentences
        """
        corpus_filepath = self.paths.corpus_filepath
        for parsed_doc in self.parse_corpus_by_line():
            for sent in parsed_doc.sents:
                line = ' '.join(lemmatize_clean(sent))
                if line != '':
                    yield line

    def parse_corpus_by_line(self, batch_size=100, n_threads=6, disable=['ner'], **kwargs):

        corpus_filepath = self.paths.corpus_filepath
        return self.nlp.pipe(read_doc_by_line(corpus_filepath),
                      batch_size=batch_size, n_threads=n_threads,
                      disable=disable, **kwargs)