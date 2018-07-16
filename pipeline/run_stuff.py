import os
print(os.getcwd())

from pipeline.preprocessing import Preprocessor
from pipeline.lda import LDABuilder
from pipeline.lda_viz import LDAViz

# NOTE: RUN FROM CONTEXTER BASE DIR!
source_name = 'yelp'

# Preprocessor uses whatever is in [source_name]/corpus_all.txt
pp = Preprocessor(source_name, spacy_model='en_core_web_lg')
pp.run_pipeline()

# LDA model uses trigram corpus by default
# n_topics = 75
# ldab = LDABuilder(source_name)
# ldab.get_lda_model(n_topics=n_topics)
# ldab.lda_pipeline()

# LDA Viz
# lda_viz = LDAViz(ldab)
# lda_viz.viz_pipeline(n_topics, 'ldaviz_{}.html'.format(n_topics))
