import os
print(os.getcwd())

from pipeline.preprocessing import Preprocessor
from pipeline.lda import LDABuilder

# NOTE: RUN FROM CONTEXTER BASE DIR!
source_name = 'all_the_news'

# Preprocessor uses whatever is in [source_name]/corpus_all.txt
pp = Preprocessor(source_name, spacy_model='en_core_web_lg')
pp.run_pipeline()

# LDA model uses trigram corpus by default
ldab = LDABuilder(source_name)
ldab.lda_pipeline()

# from gensim.models.phrases import Phrases, Phraser

# bigram_model = Phrases.load(pp.paths.bigram_model_filepath)
# Phraser(bigram_model).save(pp.paths.bigram_model_filepath)

# trigram_model = Phrases.load(pp.paths.trigram_model_filepath)
# Phraser(trigram_model).save(pp.paths.trigram_model_filepath)
