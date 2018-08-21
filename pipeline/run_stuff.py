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
# ldab.lda_pipeline()
# n_topics = 100
ldab = LDABuilder(source_name)
trigram_dictionary = ldab.get_corpus_dict(from_scratch=False)
bow_corpus = ldab.get_trigram_bow_corpus(trigram_dictionary, from_scratch=False)
# lda = ldab.get_lda_model(n_topics=n_topics)
# ldab.get_similarity_index(bow_corpus, lda)

# n_topics = 50
# lda = ldab.get_lda_model(n_topics=n_topics)
# ldab.get_similarity_index(bow_corpus, lda)

# LDA Viz
# n_topics = 100
lda_viz = LDAViz(ldab)
# lda_viz.viz_pipeline(n_topics, 'ldaviz_{}.html'.format(n_topics))
for n_topics in [50,75,100]:
	print('writing data for n_topics: {}'.format(n_topics) )
	lda = ldab.get_model(n_topics=n_topics)
	ldaviz_model = lda_viz.get_ldaviz_model(lda, bow_corpus, trigram_dictionary)
	lda_viz.write_json_data(ldaviz_model, n_topics)
