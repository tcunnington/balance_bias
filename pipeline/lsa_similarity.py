
from pipeline.similarity_model import SimilarityModel
from pipeline.lsa import LSABuilder

class LSASimilarity(SimilarityModel):

    def __init__(self, lsa_builder: LSABuilder, n_topics, trigram_dictionary):
        self.model = lsa_builder.get_lsa_model(n_topics, from_scratch=False)
        self.index = lsa_builder.get_similarity_index(trigram_dictionary, self.model, from_scratch=False)


    def get_similarity_to_doc(self, doc_bow, ntop=100):
        lsa_vec = self.model[doc_bow]
        topics = []
        sims = self.index[lsa_vec]
        return sorted(enumerate(sims), key=lambda item: -item[1])[:ntop], topics