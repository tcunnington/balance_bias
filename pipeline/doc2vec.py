
from gensim.models.doc2vec import Doc2Vec, TaggedDocument




class Doc2vecBuilder():

    def __init__(self, paths, lda_builder):
        self.paths = paths
        pass

    def pipeline(self, docs, workers=6):
        file_name = self.paths#asflkjhalskj
        documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(docs)]
        # build_vocab_from_freq # <- might do this first from doc2bow to cut down on train time?
        model = Doc2Vec(documents, vector_size=5, window=2, min_count=2, workers=workers)
        model.save(file_name)
        model = Doc2Vec.load(file_name)
        # model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
        inferred_vector = model.infer_vector('line_sentence_doc')
        recommendations = model.docvecs.most_similar([inferred_vector], topn=10)