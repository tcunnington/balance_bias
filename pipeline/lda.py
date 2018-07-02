from gensim.corpora import Dictionary, MmCorpus
from gensim.models.ldamulticore import LdaMulticore

import pyLDAvis
import pyLDAvis.gensim
import warnings
import cPickle as pickle


def lda():
	n_workers = 6
	n_topics  = 50

	trigram_dictionary = Dictionary.load(trigram_dictionary_filepath)
	trigram_bow_corpus = MmCorpus(trigram_bow_filepath)

	paths = Paths('all_the_news')
	%%time

	# this is a bit time consuming - make the if statement True
	# if you want to train the LDA model yourself.
	if 1 == 1:

	    with warnings.catch_warnings():
	        warnings.simplefilter('ignore')
	        
	        # workers => sets the parallelism, and should be
	        # set to your number of physical cores minus one
	        lda = LdaMulticore(trigram_bow_corpus,
	                           num_topics=n_topics,
	                           id2word=trigram_dictionary,
	                           workers=n_workers)
	    
	    lda.save(paths.lda_model_filepath)
	    
	# load the finished LDA model from disk
	# lda = LdaMulticore.load(paths.lda_model_filepath)