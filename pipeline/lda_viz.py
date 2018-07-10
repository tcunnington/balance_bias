import os
import pickle

import pyLDAvis
import pyLDAvis.gensim

from pipeline.lda import *
from pipeline.utils import Paths

"""
NOTE this stuff is in a separate file because loading pyLDAvis causes notebook cells to show deprecation 
errors after all runs--super annoying
"""

paths = Paths('all_the_news')

def viz_pipeline(html_filename):
    trigram_dictionary = get_corpus_dict()
    bow = get_trigram_bow_corpus(trigram_dictionary)
    lda = get_lda_model() # will just save for use later
    get_ldaviz_model(lda, bow, trigram_dictionary) # will just save for use later
    print('Creating HTML at ' + html_filename)

def get_ldaviz_model(lda_model, trigram_bow_corpus, trigram_dictionary, recalculate=False):

    if not os.path.isfile(paths.ldavis_data_filepath) or recalculate:
        LDAvis_prepared = pyLDAvis.gensim.prepare(lda_model, trigram_bow_corpus,
                                                  trigram_dictionary)

        with open(paths.ldavis_data_filepath, 'wb') as f:
            pickle.dump(LDAvis_prepared, f)
    else:
        with open(paths.ldavis_data_filepath, 'rb') as f:
            LDAvis_prepared = pickle.load(f)

    return LDAvis_prepared
