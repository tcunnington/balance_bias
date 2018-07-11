import os
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

from pipeline.paths import Paths

paths = Paths('all_the_news')

def get_word2vec_model(recalculate=False, n_epochs=11):

    n_workers = 6

    if not os.path.isfile(paths.trigram_dictionary_filepath) or recalculate:
        sentence_count = 1371601 # TODO WHAT???

        trigram_sentences = LineSentence(paths.trigram_sentences_filepath)

        # initiate the model and perform the first epoch of training
        word2vec = Word2Vec(trigram_sentences, size=100, window=5,
                            min_count=20, sg=1, workers=n_workers)

        word2vec.save(paths.word2vec_filepath)

        # more epochs of training
        for i in range(1, n_epochs + 1):
            word2vec.train(trigram_sentences, total_examples=sentence_count, epochs=word2vec.iter)
            word2vec.save(paths.word2vec_filepath)
            u'{} training epochs so far.'.format(word2vec.train_count)

    # load the finished model from disk
    word2vec = Word2Vec.load(paths.word2vec_filepath)
    word2vec.init_sims()

    return word2vec

