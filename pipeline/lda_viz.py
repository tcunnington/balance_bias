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

    def viz_pipeline(self, n_topics, html_filename=None, recalculate=True):
        if html_filename is None:
            html_filename = self.paths.get_ldaviz_filepath(n_topics)

        trigram_dictionary = self.lda_builder.get_corpus_dict(from_scratch=False)
        bow = self.lda_builder.get_trigram_bow_corpus(trigram_dictionary, from_scratch=False)
        lda = self.lda_builder.get_lda_model(n_topics=n_topics, from_scratch=False)
        ldaviz_prepared = self.get_ldaviz_model(lda, bow, trigram_dictionary, recalculate=recalculate)
        out_path = self.paths.output_file(html_filename)
        pyLDAvis.save_html(ldaviz_prepared, out_path)
        print('Creating HTML at ' + out_path)

    def get_ldaviz_model(self, lda_model, trigram_bow_corpus, trigram_dictionary, recalculate=False):

        filepath = self.paths.get_ldaviz_filepath(lda_model.num_topics)
        if not os.path.isfile(filepath) or recalculate:
            print('Building LDA Viz model...')
            LDAvis_prepared = pyLDAvis.gensim.prepare(lda_model, trigram_bow_corpus,
                                                      trigram_dictionary)

            with open(filepath, 'wb') as f:
                pickle.dump(LDAvis_prepared, f)
            print('Pickled at ' + filepath)
        else:
            print('Loading LDA Viz model...')
            with open(filepath, 'rb') as f:
                LDAvis_prepared = pickle.load(f)

        return LDAvis_prepared

    def write_json_data(self, ldaviz_model, n_topics):
        pyLDAvis.save_json(ldaviz_model, self.paths.ldaviz_json(n_topics))
