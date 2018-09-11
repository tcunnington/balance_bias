
## Overview

#### Motivation
Today most people get their news from social media or other feeds that curate what you see. Your personal recommendations are built to prioritize either content your friends like, or content similar to what you like. This automatic curation leads to what is commonly called a "filter bubble", a phenomenon where you only come in contact with information you are already inclined to agree with. With everyone living in their own information universe if can become difficult to relate to people in vastly different ones. Do a quick google search if you want more details on why what is a bad thing.

As an answer to this phenomenon I create BalanceBias: a new recommender with an ideological twist!

#### The application

BalanceBias uses natural language processing (NLP) techniques to recommend news articles that a user wouldn’t normally see. Starting with the URL for article they like, it first scrapes the content from the web, then uses topic modeling to find articles with similar content but a different “bias”. The topic model, built with Latent Dirichlet allocation, provides a means to summarize and compare articles together.

## NLP Pipeline

#### Preprocessing and phrase identification

Before doing topic modeling I first discovered common bigram and trigrams. To prepare the data for phrase identification a few preprocessing steps were necessary:
1. Segment each document into sentences
2. Lemmatize words using spaCy
3. Remove punctuation and excess spaces

When this was done I could use Gensim's Phraser class to automatically find pairs of unigrams that occurred more commonly than would be expected by chance. These were grouped together for all subsequent steps.
I then ran the phrase model again to discover common trigrams as well.

#### Topic modeling using latent Dirichlet allocation

After identifying phrases up to trigrams I used Gensim's LDAMulticore class to build a topic model. Various values of k, the number of topics, were chosen. According to topic coherence the best value was around 200 topics. However I wanted the model to be sensitive to very specific topics--instead of recommending another article on "politics" for instance, a user would likely want to see another article on "the renegotiation of NAFTA". For this reason I chose to use a value above that the optimal one chosen by topic coherence. Perceived performance improved as a result. 

## Future

An improved bias model is the main direction for future work. Right now the bias is based on the publication it came from, where values where hand picked to conform to this source: 
I will either hand label by data (again by source) to train a new model, or else use an existing data set. Data sets for building fake news detectors look promising.