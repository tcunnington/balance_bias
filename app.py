from flask import Flask, render_template, request, redirect, url_for, session
from pipeline.preprocessing import Preprocessor
from pipeline.lda import LDABuilder
# from pipeline.lda_similarity import LDASimilarity
# from pipeline.corpus import Corpus
from article_scraper import ArticleScraper
from newspaper.article import ArticleException

from news_api import NewsAPI
from recommendation_page import RecommendationPage

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
# os.getenv('NEWS_API_KEY')

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

#### PARAMS ####
source = 'all_the_news'
n_topics = 200
n_stories = 10
n_chunk = 500
is_lsa = False

#### Objects needed in memory for recommendations ####
prep = Preprocessor(preload_models=True)
# corpus = Corpus()
rec_page = RecommendationPage(app)

lda_builder = LDABuilder()
trigram_dictionary = lda_builder.get_corpus_dict(from_scratch=False)


lda = lda_builder.get_model(n_topics, from_scratch=False)
news_api = NewsAPI(lda)

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
            parsed_doc = prep.process_doc(title + ' ' + text)
        except ArticleException as e:
            # If the download for some reason fails (ex. 404) we need to show an error msg and redirect to main
            print('SCRAPING FAILED!')
            return redirect(url_for('.index', scraping_error=True)) # TODO pass this without adding to url params (ugly)
    else:
        title = 'Raw text: ' + text[:50]
        parsed_doc = prep.process_doc(text)

    bow = trigram_dictionary.doc2bow(parsed_doc)
    # sims, topics = similarity_model.get_similarity_to_doc(bow)

    # similarity_model.hellinger_distance(bow, lda_builder.get_trigram_bow_corpus(trigram_dictionary))

    # [idxs, cos_scores] = list(zip(*sims))

    # TODO move it all to rec page -> just return the display data

    # find source and get valid biases
    source_name = rec_page.resolve_source_name(url)
    source_id = rec_page.resolve_source_id(source_name)
    bias_code = rec_page.resolve_bias_code(source_id)
    valid_biases = rec_page.resolve_valid_biases(bias_code)

    # make query
    doc_topics = sorted(lda.get_document_topics(bow, minimum_probability=0.05), key=lambda x: -x[1])
    topics = [{'id': tid, 'words': [w for w, pw in lda.show_topic(tid, 5)]} for tid, p in doc_topics]
    valid_sources = rec_page.get_valid_sources(valid_biases)
    rdf = news_api.query(doc_topics, valid_sources, parsed_doc)

    # get display info from meta data
    # rdf = corpus.meta_data.loc[list(idxs)]

    # Bias filtering
    rdf = rec_page.append_bias(rdf)
    rdf = rdf[rdf['bias'].isin(valid_biases)] # add title includes words from top topics
    # fields = ['title', 'publication', 'url', 'description', 'bias_c', 'bias_label','icon_url']

    render_data = {
        'source_icon': rec_page.resolve_img_url(source_id),
        'headline': title,
        'text_snippet': text[:400] + '...',
        'topics': topics,
        'bias_display': rec_page.resolve_bias_display(source_id),
        'bias_code': bias_code,
        'recommendations': rec_page.build_recommendations_list(rdf, [], n_stories)
    }
    return render_template('recommendations.html', data=render_data)



if __name__ == '__main__':
  app.run(host='0.0.0.0', port=33507)


# Postgres:
# from flask.ext.sqlalchemy import SQLAlchemy
# app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://localhost/${whoami}'
# db = SQLAlchemy(app)
