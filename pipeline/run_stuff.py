from preprocessing import *


preprocessing_pipeline('all_the_news')


# print('Loading spaCy..')
# nlp = spacy.load('en')
# bigram_model = get_bigram_model()
# trigram_model = get_trigram_model()
# # print('Writing tri-gram sentences..')
# # write_trigram_sentences(trigram_model)
# print('Writing tri-gram reviews..')
# write_trigram_reviews(nlp, bigram_model, trigram_model)
# print('Done done!')