from pipeline.similarity_model import SimilarityModel
from pipeline.lda import LDABuilder
from gensim.matutils import kullback_leibler, jaccard, hellinger

class LDASimilarity(SimilarityModel):

    def __init__(self, lda_builder: LDABuilder, n_topics, trigram_dictionary):
        self.model = lda_builder.get_model(n_topics, from_scratch=False)
        self.similarity_index = lda_builder.get_similarity_index(trigram_dictionary, self.model, from_scratch=False)


    def get_similarity_to_doc(self, doc_bow, ntop=100):
        doc_topics = self.model.get_document_topics(doc_bow, minimum_probability=0.05)
        topics = [{'id': tid, 'words':[w for w,pw in self.model.show_topic(tid, 5)]} for tid,p in doc_topics] # TODO
        sims = self.similarity_index[doc_topics]
        # sort by cos distance and take top chunk since there's no reason to carry them all around.
        return sorted(enumerate(sims), key=lambda item: -item[1])[:ntop], topics

    def hellinger_distance(self, doc_bow, trigram_dictionary):
        corpus = self.model.get_trigram_bow_corpus(trigram_dictionary, from_scratch=False)
        heap = []
        for document in corpus:
            print(hellinger(doc_bow, document))