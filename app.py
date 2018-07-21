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
    url = request.form.get('urlInput')
    text = request.form.get('textInput')

    # if url is not None:
    #     parsed_doc = scrape(url)
    # else:
    #     parsed_doc = prep.process_doc(text)
    #
    # bow = lda_builder.trigram_doc_to_bow(parsed_doc)
    # topic_ids = lda_builder.choose_topics_subset(lda[bow])

    source_icon = '/static/imgs/cnn.jpg'
    headline = 'Michael Cohen Secretly Taped Trump Discussing Payment to Playboy Model'
    topic_names = ['Russia', 'Cohen', 'Collusion', 'White House', 'Election']
    bias_score = 4.5
    recommendations = [{'headline':'YAAAY', 'content':'aklsdjfhak Discussing ljsdfh lad  blahrh asdkjh', 'source':'WP', 'bias':3} for i in range(5)]

    render_data = {
        'source_icon': source_icon,
        'headline': headline,
        'topics': topic_names,
        'bias': bias_score,
        'recommendations': recommendations
    }
    return render_template('recommendations.html', data=render_data)

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
