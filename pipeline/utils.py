import codecs


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


def punct_space(token):
    """
    helper function to eliminate tokens
    that are pure punctuation or whitespace
    """

    return token.is_punct or token.is_space


def line_review(filename):
    """
    generator function to read in reviews from the file
    and un-escape the original line breaks in the text
    """

    with codecs.open(filename, encoding='utf_8') as f:
        for review in f:
            yield review.replace('\\n', '\n')


def lemmatized_sentence_corpus(nlp_model, filename):
    """
    generator function to use spaCy to parse reviews,
    lemmatize the text, and yield sentences
    """

    for parsed_review in nlp_model.pipe(line_review(filename),
                                        batch_size=100, n_threads=6, disable=['tagger', 'ner']):

        for sent in parsed_review.sents:
            yield u' '.join([token.lemma_ for token in sent if not punct_space(token)])


def add_newline(itr):
    """
    generator wrapper to add newlines to items of an iterator
    """
    for item in itr:
        yield item + '\n'
