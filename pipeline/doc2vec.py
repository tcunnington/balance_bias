import os

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.word2vec import LineSentence



class Doc2vecBuilder():

    def __init__(self, paths):
        self.paths = paths

    def pipeline(self):
        self.get_model(ndim=50, recalculate=True)

    def get_model(self, ndim=50, n_workers=6, recalculate=False, from_scratch=True):

        filepath = self.paths.get_doc2vec(ndim)

        if not os.path.isfile(filepath) or recalculate:

            if not from_scratch:
                raise ValueError('No doc2vec file exists but from_scratch is False')

            # build_vocab_from_freq # <- might do this first from doc2bow to cut down on train time?

            print('Building doc2vec model...')
            trigram_docs = LineSentence(self.paths.trigram_corpus_filepath)
            documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(trigram_docs)]
            model = Doc2Vec(documents, vector_size=50, min_count=2, workers=n_workers)  # window=?
            # model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

            model.save(filepath)
            print('doc2vec model (ndims={}) written to {}'.format(ndim, filepath))
        else:
            print('Loading doc2vec model (n_topics={})...'.format(ndim))
            model = Doc2Vec.load(filepath)

        return model

    def get_similarity_to_doc(self, model, doc):
        inferred_vector = model.infer_vector(doc) # as line sentences array
        return model.docvecs.most_similar([inferred_vector], topn=10)