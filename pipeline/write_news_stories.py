import pandas as pd
from paths import Paths
import spacy

import os
import fnmatch

paths = Paths()

nlp = spacy.load('en')
file_name = 'articles*'

def prep_whitespace(text):
    return re.sub(r'\s+', ' ', text).strip() # .replace('\n', '\\n') # instead of simply removing all newlines..

def story_content(file_pattern):

    for file in os.listdir(paths.data_directory):
        if fnmatch.fnmatch(file, file_pattern):
            print(file)
            content = pd.read_csv(paths.data_file(file))['content']

            for story in content:
                yield prep_whitespace(story)

def partial_text(text):
    text = re.sub("[‹»›„‚“‟‘‛”’\"❛❜❝❞❮❯〝〞 ]{2,}", " ", text)
    return prep_whitespace(text)[:500]

def write_meta_data():
    """
    Save everything that is not the story content to a separate file
    :return:
    """
    dfs = []
    for fp in [paths.data_file('articles' + str(i) + '.csv') for i in [3, 1, 2]]:
        print(fp)
        dfs.append(pd.read_csv(fp))

    df = pd.concat(dfs).reset_index(drop=True)
    df.drop(columns=['Unnamed: 0'], inplace=True)

    df['partial_content'] = df['content'].map(lambda x: partial_text(x))
    df.drop(columns=['content'], inplace=True)
    df.to_csv(paths.corpus_meta_data)


if __name__ == "__main__":
    write_meta_data()
    # batch_write(paths.corpus_filepath, add_newline(story_content(file_name)), batch_size=1000)
