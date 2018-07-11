import pandas as pd
from pipeline.utils import *
from pipeline.preprocessing import Paths
import spacy

import os
import fnmatch

paths = Paths('all_the_news')

nlp = spacy.load('en')
file_name = 'articles*'

for file in os.listdir(paths.data_directory):
    if fnmatch.fnmatch(file, file_name):
        print(file)
        df = pd.read_csv(file)
        ## TODO -> TEST
        # .replace('\n', '\\n') # instead of simply removing all newlines..
        itr = (prep_whitespace(x) for x in df['content'])
        batch_write(paths.corpus_filepath, add_newline(itr))

# def get_sentences(texts):
#     for doc in nlp.pipe(texts, batch_size=100, n_threads=4):
#         for sent in doc.sents:
#             s = ' '.join([token.lemma_ for token in sent if not (token.is_punct or token.is_space)])
#             if(len(s.replace(' ',''))!=0): # Dunno wtf is going on here. Looks like shitty data
#                 # f.write(s)
#                 yield s
