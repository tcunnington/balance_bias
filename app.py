from flask import Flask, render_template, request, redirect, url_for, session
from pipeline.preprocessing import Preprocessor
from pipeline.lda import LDABuilder
from pipeline.lsa import LSABuilder
from pipeline.corpus import Corpus
from article_scraper import ArticleScraper
from newspaper.article import ArticleException

from recommendation_page import RecommendationPage

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

#### PARAMS ####
source = 'all_the_news'
n_topics = 100
n_stories = 10
n_chunk = 500
is_lsa = False

#### Objects needed in memory for recommendations ####
prep = Preprocessor(source, preload_models=True)
lda_builder = LDABuilder(source)
lda = lda_builder.get_lda_model(n_topics, from_scratch=False)
trigram_dictionary = lda_builder.get_corpus_dict(from_scratch=False)
# LSA
lsa_builder = LSABuilder(source, lda_builder)
lsa = lsa_builder.get_lsa_model(n_topics, from_scratch=False)
similarity_index = lda_builder.get_similarity_index(trigram_dictionary, lda, from_scratch=False)
lsa_index = lsa_builder.get_similarity_index(trigram_dictionary, lsa, from_scratch=False)

corpus = Corpus(source)
rec_page = RecommendationPage(app)

@app.route('/')
def index():
    scraping_error = request.args.get('scraping_error', False)
    return render_template('main.html', scraping_error=scraping_error)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/presentation')
def presentation():
    return render_template('presentation.html')

@app.route('/recommendations', methods=['POST'])
def recommendations():
    url = request.form.get('urlInput', '')
    text = request.form.get('textInput', '')

    if url != '':
        try:
            title, text = ArticleScraper.scrape(url)
            parsed_doc = prep.process_doc(text)
        except ArticleException as e:
            # If the download for some reason fails (ex. 404) we need to show an error msg and redirect to main
            print('SCRAPING FAILED!')
            return redirect(url_for('.index', scraping_error=True)) # TODO pass this without adding to url params (ugly)
    else:
        title = 'Raw text: ' + text[:50]
        parsed_doc = prep.process_doc(text)

    bow = trigram_dictionary.doc2bow(parsed_doc)

    if is_lsa:
        lsa_vec = lsa[bow]
        topics = []
        sims = lsa_index[lsa_vec]
        sims = sorted(enumerate(sims), key=lambda item: -item[1])[:n_chunk]
    else:
        doc_topics = lda.get_document_topics(bow, minimum_probability=0.05)
        topics = [{'id': tid, 'words':[w for w,pw in lda.show_topic(tid, 5)]} for tid,p in doc_topics] # TODO
        sims = similarity_index[doc_topics]
        # sort by cos distance and take top chunk since there's no reason to carry them all around.
        sims = sorted(enumerate(sims), key=lambda item: -item[1])[:n_chunk]

    [idxs, cos_scores] = list(zip(*sims))

    # get display info from meta data
    rdf = corpus.meta_data.loc[list(idxs)]

    source_name = rec_page.resolve_source_name(url)
    source_id = rec_page.resolve_source_id(source_name)
    source_icon = rec_page.resolve_img_url(source_id)
    bias_code = rec_page.resolve_bias_code(source_name)
    bias_display = rec_page.resolve_bias_display(source_name)

    # Bias filtering
    bias_filter = rec_page.resolve_valid_biases(bias_code)
    rdf = rec_page.append_bias(rdf)
    rdf = rdf[rdf['bias'].isin(bias_filter)] # add title includes words from top topics
    fields = ['title', 'publication', 'url', 'partial_content', 'bias', 'bias_label','icon_url']

    render_data = {
        'source_icon': source_icon,
        'headline': title,
        'text_snippet': text[:400] + '...',
        'topics': topics,
        'bias_display': bias_display,
        'bias_code': bias_code,
        'recommendations': rec_page.build_recommendations_list(rdf, fields, n_stories)
    }
    return render_template('recommendations.html', data=render_data)



if __name__ == '__main__':
  app.run(port=33507)


# Postgres:
# from flask.ext.sqlalchemy import SQLAlchemy
# app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://localhost/${whoami}'
# db = SQLAlchemy(app)
