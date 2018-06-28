import os

if not os.path.isfile(unigram_sentences_filepath):
    write_unigram_sentences()
else:
    print('What the fuck are you doing!?!?! Are you sure you want to overwrite that file?? If so delete it first.')

