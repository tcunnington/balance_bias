import codecs
import re


def batch_write(wfilename, iterator, batch_size=100):
    """
    Write files by the batch from an iterator. Created under the (possibly mistaken) belief that it's faster than
    asking for many small writes
    """
    count = 0
    with codecs.open(wfilename, 'w', encoding='utf_8') as f:
        batch = []
        for item in iterator:
            batch.append(item)
            count += 1

            if len(batch) == batch_size:
                f.write(''.join(batch))
                batch = []

        # final write for partial patch
        f.write(''.join(batch))
        print(u'''Wrote {:,} items to the new txt file '{}'.'''.format(count, wfilename))

def add_newline(itr):
    """
    generator wrapper to add newlines to items of an iterator
    """
    for item in itr:
        yield item + '\n'

def prep_whitespace(text):
    return re.sub(r'\s+', ' ', text)


#################################################
#
#     Misc  / spaCy
#
#################################################


def is_punct_space(token):
    """
    helper function to find tokens that are pure punctuation or whitespace
    """
    return token.is_punct or token.is_space

def read_doc_by_line(filename):
    """
    generator to read in a file one line at a time, and un-escape the original line breaks in the text
    """
    with codecs.open(filename, encoding='utf_8') as f:
        for doc in f:
            yield doc.replace('\\n', '\n') # TODO remove?

def lemmatize_clean(spacy_itr):
    """
    generator to act on a spaCy object that allows iterating over tokens
    """
    return (token.lemma_ for token in spacy_itr if not is_punct_space(token))