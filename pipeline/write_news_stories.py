import pandas as pd
from pipeline.utils import *
from pipeline.paths import Paths
import spacy

import os
import fnmatch

paths = Paths('all_the_news')

nlp = spacy.load('en')
file_name = 'articles*'

def story_content(file_pattern):

	for file in os.listdir(paths.data_directory):
	    if fnmatch.fnmatch(file, file_pattern):
	        print(file)
	        content = pd.read_csv(paths.data_file(file))['content']
	        ## TODO -> TEST
	        # .replace('\n', '\\n') # instead of simply removing all newlines..
	        for story in content:
	        	yield prep_whitespace(story)
	        
batch_write(paths.corpus_filepath, add_newline(story_content(file_name)), batch_size=1000)

# def get_sentences(texts):
#     for doc in nlp.pipe(texts, batch_size=100, n_threads=4):
#         for sent in doc.sents:
#             s = ' '.join([token.lemma_ for token in sent if not (token.is_punct or token.is_space)])
#             if(len(s.replace(' ',''))!=0): # Dunno wtf is going on here. Looks like shitty data
#                 # f.write(s)
#                 yield s
