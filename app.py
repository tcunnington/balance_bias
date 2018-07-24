from flask import Flask, render_template, request, redirect
from pipeline.preprocessing import Preprocessor
from pipeline.lda import LDABuilder
from pipeline.corpus import Corpus
from article_scraper import ArticleScraper
from newspaper.article import ArticleException
from sources import Sources

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

#
# Objects needed in memory for recommendations
#

source = 'all_the_news'
n_topics = 75
corpus = Corpus(source)
prep = Preprocessor(source, preload_models=True)
lda_builder = LDABuilder(source)
lda = lda_builder.get_lda_model(n_topics, from_scratch=False)
trigram_dictionary = lda_builder.get_corpus_dict(from_scratch=False)
corpus_topics_matrix = lda_builder.get_corpus_topics_matrix(n_topics, from_scratch=False)
sources = Sources()

@app.route('/')
def index():
    return render_template('main.html')

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
            print('Trying to scrape')
            title, text = ArticleScraper.scrape(url)
            # print(title, text)
            parsed_doc = prep.process_doc(text)
        except ArticleException as e:
            # TODO If the download for some reason fails (ex. 404) we need to show an error msg and redirect to main
            print('SCRAPING FAILED!')
            redirect('/')
            return
    else:
        print('Parsing raw input')
        title = 'Raw text: ' + text[:50]
        parsed_doc = prep.process_doc(text)

    bow = trigram_dictionary.doc2bow(parsed_doc)
    doc_topics = lda[bow]

    # topics display names
    topic_ids = lda_builder.choose_topics_subset(doc_topics)
    print(topic_ids)

    # recommendations
    topic_vec = lda_builder.create_topic_vec(lda.num_topics, doc_topics)
    closest_idxs = lda_builder.cosine_similarity_corpus(topic_vec, corpus_topics_matrix)
    rdf = corpus.meta_data.loc[closest_idxs]
    recommendations = [row.to_dict() for i, row in rdf[['title', 'publication','url','partial_content']].iterrows()]
    # print(rdf)


    source_icon = '/static/imgs/cnn.jpg'
    # headline = 'Michael Cohen Secretly Taped Trump Discussing Payment to Playboy Model'
    topic_names = ['Russia', 'Cohen', 'Collusion', 'White House', 'Election'] # TODO topic_ids
    source_name = 'The New York Times' # TODO get from url
    bias_score = sources.sources_bias_map[source_name] # TODO display name

    render_data = {
        'source_icon': source_icon,
        'headline': title,
        'topics': topic_names,
        'bias': bias_score,
        'recommendations': recommendations
    }
    return render_template('recommendations.html', data=render_data)



if __name__ == '__main__':
  app.run(port=33507)


# Postgres:
# from flask.ext.sqlalchemy import SQLAlchemy
# app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://localhost/${whoami}'
# db = SQLAlchemy(app)
