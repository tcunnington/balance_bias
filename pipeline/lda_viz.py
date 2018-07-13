import os
import pickle

import pyLDAvis
import pyLDAvis.gensim

from pipeline.lda import LDABuilder

"""
NOTE this stuff is in a separate file because loading pyLDAvis causes notebook cells to show deprecation 
errors after all runs--super annoying
"""

class LDAViz:

    def __init__(self, lda_builder: LDABuilder):
        self.lda_builder = lda_builder
        self.paths = lda_builder.paths

    def viz_pipeline(self, n_topics, html_filename='lda_viz.html', recalculate=True):
        trigram_dictionary = self.lda_builder.get_corpus_dict(from_scratch=False)
        bow = self.lda_builder.get_trigram_bow_corpus(trigram_dictionary, from_scratch=False)
        lda = self.lda_builder.get_lda_model(n_topics=n_topics, from_scratch=False)
        ldaviz_prepared = self.get_ldaviz_model(lda, bow, trigram_dictionary, recalculate=recalculate)
        out_path = self.paths.output_file(html_filename)
        pyLDAvis.save_html(ldaviz_prepared, out_path)
        print('Creating HTML at ' + out_path)

    def get_ldaviz_model(self, lda_model, trigram_bow_corpus, trigram_dictionary, recalculate=False):

        if not os.path.isfile(self.paths.get_ldaviz_filepath(lda_model.num_topics)) or recalculate:
            LDAvis_prepared = pyLDAvis.gensim.prepare(lda_model, trigram_bow_corpus,
                                                      trigram_dictionary)

            with open(self.paths.ldavis_data_filepath, 'wb') as f:
                pickle.dump(LDAvis_prepared, f)
        else:
            with open(self.paths.ldavis_data_filepath, 'rb') as f:
                LDAvis_prepared = pickle.load(f)

        return LDAvis_prepared
