import os
print(os.getcwd())

from pipeline.preprocessing import Preprocessor
from pipeline.lda import LDABuilder
from pipeline.lda_viz import LDAViz

# NOTE: RUN FROM CONTEXTER BASE DIR!
source_name = 'all_the_news'

# Preprocessor uses whatever is in [source_name]/corpus_all.txt
# pp = Preprocessor(source_name, spacy_model='en_core_web_lg')
# pp.run_pipeline()

# LDA model uses trigram corpus by default
n_topics = 75
ldab = LDABuilder(source_name)
lda = ldab.get_lda_model(n_topics=n_topics)
# ldab.lda_pipeline()

trigram_dictionary = ldab.get_corpus_dict(from_scratch=False)
bow_corpus = ldab.get_trigram_bow_corpus(trigram_dictionary, from_scratch=False)
ldab.get_similarity_index(bow_corpus, lda)

# LDA Viz
# lda_viz = LDAViz(ldab)
# lda_viz.viz_pipeline(n_topics, 'ldaviz_{}.html'.format(n_topics))
