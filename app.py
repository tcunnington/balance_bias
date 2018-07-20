# import datetime
# import itertools
# import os
from flask import Flask, render_template, request, redirect
# import requests as req
# import json
# import pandas as pd
# from bokeh.plotting import figure
# from bokeh.embed import components
# from bokeh.palettes import Category10 as palette

from pipeline.preprocessing import Preprocessor
from pipeline.lda import LDABuilder

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True


source = 'all_the_news'
# prep = Preprocessor(source, preload_models=True)
#
# lda_builder = LDABuilder(source)
# lda = lda_builder.get_lda_model()

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/recommendations', methods=['POST'])
def context():
    print(list(request.form.keys()))
    article = request.form.get('article')
    # features = request.form.getlist('features')

    # raw_html = spacy.displacy.render(prep.nlp(article), style='ent')

    # parsed_doc = prep.process_doc(article)
    # bow = lda_builder.trigram_doc_to_bow(parsed_doc)
    # topic_ids = lda_builder.choose_topics_subset(lda[bow])
    #
    # render_data = {
    #     'article':raw_html,
    #     'topics': topic_ids
    # }
    # return render_template('recommendations.html', data=render_data)

if __name__ == '__main__':
  app.run(port=33507)


# r.status_code
# r.headers['content-type']
# r.encoding
# r.text
# r.json()


# Postgres:
# from flask.ext.sqlalchemy import SQLAlchemy
# app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://localhost/${whoami}'
# db = SQLAlchemy(app)
