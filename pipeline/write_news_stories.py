import pandas as pd
from utils import *
from preprocessing import Paths
import spacy

paths = Paths('all_the_news')
df = pd.read_csv(paths.data_file('articles1.csv'))
nlp = spacy.load('en')

def get_sentences(texts):
    for doc in nlp.pipe(texts, batch_size=100, n_threads=4):
        for sent in doc.sents:
            s = ' '.join([token.lemma_ for token in sent if not (token.is_punct or token.is_space)])
            if(len(s.replace(' ',''))!=0): # Dunno wtf is going on here. Looks like shitty data
                # f.write(s)
                yield s

# itr = (prep_whitespace(x.encode("utf-8").decode("utf-8")) for x in df['content'])
itr = (prep_whitespace(x) for x in df['content'])
batch_write(paths.corpus_filepath, add_newline(itr))