import datetime
import itertools
import os
from flask import Flask, render_template, request, redirect
import requests as req
import json
import pandas as pd
from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.palettes import Category10 as palette

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

app = Flask(__name__)

FIELD_MAP = {
    'close':    'Closing price',
    'adj_close':'Adjusted closing price',
    'open':     'Opening price',
    'adj_open': 'Adjusted opening price',
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/prices', methods=['POST'])
def prices():
    quandl_api_key = os.getenv('QUANDL_API_KEY')
    date_start = '2017-01-01'
    date_end = '2018-01-01'

    ticker = request.form.get('ticker')
    features = request.form.getlist('features')

    if ticker == '':
        ticker = 'GOOG' # front end should give this

    url_params = {
        'ticker': ticker,
        'date.gte': date_start,
        'date.lte': date_end,
        'api_key': quandl_api_key
    }

    r = req.get('https://www.quandl.com/api/v3/datatables/WIKI/PRICES', params=url_params)

    if r.status_code != 200:
        raise req.RequestException('API Error (' + str(r.status_code) + '): ' + r.reason, response=r)

    price_data = r.json()['datatable']

    # To DataFrame
    colnames = [c['name'] for c in price_data['columns']]
    df = pd.DataFrame(data=price_data['data'], columns=colnames)
    df['date'] = pd.to_datetime(df["date"])

    # PLOT
    plot = figure(title="Quandl WIKI EOD Stock Prices - 2017",
                  x_axis_label='Date', x_axis_type='datetime',
                  y_axis_label='Price ($)')

    colors = itertools.cycle(palette[10])
    for feature, color in zip(features, colors):
        plot.line(df['date'], df[feature], legend=ticker + ' - ' + FIELD_MAP[feature], color=color)

    plot.legend.location = 'top_left'
    script, div = components(plot)

    # RENDER
    context = {
        'ticker': ticker,
        'bk_script': script,
        'bk_el': div,
    }

    return render_template('prices.html', d=context)

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
