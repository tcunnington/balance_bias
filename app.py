from flask import Flask, render_template, request, redirect
from pipeline.preprocessing import Preprocessor
from pipeline.lda import LDABuilder
from pipeline.corpus import Corpus
from article_scraper import ArticleScraper
from newspaper.article import ArticleException

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

#
# Objects needed in memory for recommendations
#

source = 'all_the_news'
corpus = Corpus(source)
prep = Preprocessor(source, preload_models=True)
lda_builder = LDABuilder(source)
lda = lda_builder.get_lda_model()
trigram_dictionary = lda_builder.get_corpus_dict()

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
def context():
    url = request.form.get('urlInput', '')
    text = request.form.get('textInput', '')

    # print(url, text)

    if url != '':
        try:
            print('Trying to scrape')
            title, text = ArticleScraper.scrape(url)
            parsed_doc = prep.process_doc(text)
        except ArticleException as e:
            # TODO If the download for some reason fails (ex. 404) we need to show an error msg and redirect to main
            print('SCRAPING FAILED!')
            redirect('/')
            return
    else:
        print('Parsing raw input')
        title = 'News article title....'
        parsed_doc = prep.process_doc(text)

    bow = trigram_dictionary.doc2bow(parsed_doc)
    doc_topics = lda[bow]

    # topics display names
    topic_ids = lda_builder.choose_topics_subset(doc_topics)

    # recommendations
    topic_vec = lda_builder.create_topic_vec(lda.num_topics, doc_topics)
    closest_idxs = lda_builder.cosine_similarity_corpus(topic_vec)
    rdf = corpus.meta_data.loc[closest_idxs]
    recommendations = [row.to_dict() for i, row in rdf[['title', 'publication','url','partial_content']].iterrows()]


    source_icon = '/static/imgs/cnn.jpg'
    # headline = 'Michael Cohen Secretly Taped Trump Discussing Payment to Playboy Model'
    topic_names = ['Russia', 'Cohen', 'Collusion', 'White House', 'Election'] # topic_ids
    bias_score = 4.5
    # recommendations = [{'headline':'YAAAY', 'content':'aklsdjfhak Discussing ljsdfh lad  blahrh asdkjh', 'source':'WP', 'bias':3} for i in range(5)]

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
