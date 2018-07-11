from pipeline.preprocessing import Preprocessor
from pipeline.lda import lda_pipeline

# NOTE: RUN FROM CONTEXTER BASE DIR!



# Preprocessor uses whatever is in [source_name]/corpus_all.txt
# pp = Preprocessor('all_the_news', spacy_model='en_core_web_lg')
# pp.run_pipeline()

# LDA model uses trigram corpus by default
# lda_pipeline()
