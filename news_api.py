import os
from itertools import compress
import requests
import pandas as pd

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

flatten = lambda l: [item for sublist in l for item in sublist] # TODO

BASE_URL = 'https://newsapi.org/'
EVERYTHING_ENDPOINT = '/v2/everything'
SOURCES_ENDPOINT = '/v2/sources'


# # TODO get this from Rec page
# def resolve_source_id(source_name):
#     x = source_name.lower().split()
#     return '-'.join(x)

# # TODO move this
# def get_sources_by_bias(bias, sources_filter=None):
#     """
#
#     :param bias:
#     :param sources_filter:
#     :return:
#     """
#     if not isinstance(bias, list):
#         bias = [bias]
#
#     sources = flatten([[resolve_source_id(name) for name in bias_sources_map[b]] for b in bias])
#     # filter and concat
#     sources = filter(lambda x: x in sources_filter, sources) if sources_filter is not None else sources
#     return



class NewsAPI:

    def __init__(self, lda):
        self.lda = lda

    def _call_api(self, url, params, min_articles):

        has_enough_articles = False
        while not has_enough_articles:
            print(params['q'])
            response = requests.get(url, params)
            json_resp = response.json()

            has_enough_articles = json_resp['totalResults'] > min_articles

            if len(params['q'].split()) < 3:
                break # won't be specific enough if we use too few words

            params['q'] = ' '.join(params['q'].split()[:-1])

        print(json_resp['totalResults'], params)
        return json_resp

    def _topic_words_top(self, doc_topics, words_per_topic=5):
        n_topics = 2

        top2topics = [tid for tid, p in sorted(doc_topics, key=lambda item: -item[1])[:n_topics]]
        words = []
        [words.extend(self.lda.show_topic(tid, words_per_topic)) for tid in top2topics]
        return [w for w, p in words]

    def _topic_words_joint(self, doc_topics, words_per_topic=6, include_prob=False):
        # words_per_topic should be >= to total # of search words in case there is just one topic, however unlikely
        words_joint_proba = [[(self.lda.id2word[wid], wp * p / self.lda.expElogbeta[:, wid].sum()) for wid, wp, in self.lda.get_topic_terms(tid, words_per_topic)] for
                             tid, p in doc_topics]
        words = sorted(flatten(words_joint_proba), key=lambda x: -x[1])
        return words if include_prob else [w for w, p in words]

    # TODO... figure out how you wanna filter.. and then build it in
    def _filter_query(self, words, parsed_doc):
        # look over first 50 words
        t = set([w.replace('.','')  for w in parsed_doc[:100]])
        # look for bi/trigrams as well..
        bools = [w.replace('.','') in t for w in words]
        return compress(words, bools)

    #################################################
    #
    #     Public
    #
    #################################################

    def query(self, doc_topics, sources, parsed_doc):
        """
        API returns articles with columns:
        'author', 'content', 'description', 'publishedAt', 'source', 'title', 'url', 'urlToImage'
        :return:
        """
        url = BASE_URL + EVERYTHING_ENDPOINT
        # top_topic_words = self._topic_words_joint(doc_topics)
        top_topic_words = self._topic_words_top(doc_topics)

        key_params = {
            'apiKey': os.getenv('NEWS_API_KEY'),
        }

        other_params = {
            'q': self.build_query(top_topic_words),# self.build_query(self._filter_query(top_topic_words, parsed_doc)),
            'sources': ','.join(sources),
            'language': 'en',
        }

        params = {**other_params, **key_params}

        n_articles = 10
        json_resp = self._call_api(url, params, n_articles)
        # response = requests.get(url, params)
        # json_resp = response.json()

        # get titles and content previews

        articles = pd.DataFrame(json_resp['articles'][:n_articles])
        articles['content'] = articles['content'].apply(lambda x: x.split('…')[0] + '…') # # TODO display concern- do elsewhere
        return articles


    @staticmethod
    def build_query(words):
        as_unigrams = ['"' + w.replace('_', ' ') + '"' if '_' in w else w for w in words]
        return ' '.join(list(set(as_unigrams))) # dedup


if __name__ == '__main__':
    n = NewsAPI({})