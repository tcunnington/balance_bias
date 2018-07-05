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

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

app = Flask(__name__)

# FIELD_MAP = {
#     'close':    'Closing price',
#     'adj_close':'Adjusted closing price',
#     'open':     'Opening price',
#     'adj_open': 'Adjusted opening price',
# }

import spacy
nlp = spacy.load('en')


@app.route('/')
def index():
    return render_template('main.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/context', methods=['POST'])
def context():
    article = request.form.get('article')
    # features = request.form.getlist('features')

    doc = nlp(article)
    raw_html = spacy.displacy.render(doc, style='ent')

    render_data = {
        'article':raw_html
    }
    return render_template('context.html', data=render_data)

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
