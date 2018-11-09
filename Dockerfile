FROM continuumio/miniconda3

ADD ./requirements.txt /tmp/requirements.txt
ADD ./conda-requirements.txt /tmp/conda-requirements.txt

RUN pip install -qr /tmp/requirements.txt
RUN conda install --yes --file /tmp/conda-requirements.txt

ADD news_api.py /opt/
ADD recommendation_page.py /opt/
ADD sources.py /opt/
ADD app.py /opt/
ADD article_scraper.py /opt/
ADD pipeline/preprocessing.py /opt/pipeline/
ADD pipeline/prep_utils.py /opt/pipeline/
ADD pipeline/paths.py /opt/pipeline/
ADD pipeline/lda.py /opt/pipeline/
ADD pipeline/models/lda/lda.n200* /opt/pipeline/models/lda/
ADD pipeline/models/trigram_dict_all.dict /opt/pipeline/models/
ADD pipeline/models/trigram_model_all /opt/pipeline/models/
ADD pipeline/models/bigram_model_all /opt/pipeline/models/
ADD static/ /opt/static/
ADD templates/ /opt/templates/

WORKDIR /opt/

RUN python -m spacy download en

EXPOSE 33507

# use $PORT somehow?
CMD gunicorn --bind 0.0.0.0:33507 wsgi
CMD python app.py


# docker images -q | xargs docker rmi -f
# docker run -it -p 33507:33507 --name test myapp
# docker build -t myapp .