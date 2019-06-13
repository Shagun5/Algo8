from typing import Any

import numpy as np
import logging
from nltk.corpus import stopwords
import pandas as pd
from gensim.models import LdaModel, LdaMulticore
from gensim import corpora
import gensim.downloader as api
from gensim.utils import simple_preprocess, lemmatize
from nltk.corpus import stopwords
from gensim import similarities
import matplotlib.pyplot as pl
import re

import time
start_time = time.time()

logging.basicConfig(format='%(asctime)%: %(levelname)% : %(message)%, level = logging.INFO')

excel_lines = pd.read_excel("tdnew.xlsx", sheet_name ='Sheet3')
excel_dict = excel_lines.to_dict(orient='records')
excel_list = [x['tweet'] for x in excel_dict if isinstance(x['tweet'], str)]

def clean_data(raw_corpus):
    final_lines = []
    for line in raw_corpus:
        line_split = line.split()
        interm_line = []
        for word in line_split:
            if word.startswith("@"):
                continue
            if word.startswith("#"):
                continue
            # Add more criteria to clean data
            interm_line.append(word)
        final_lines.append(' '.join(interm_line))
    return final_lines

raw_corpus = clean_data(excel_list)

# Create a set of frequent words
stoplist = set('that to is if I let RT @ at rt for from his her of and a the if what where who I you they we !!! this on'.split(' '))
print(stoplist)
stop = set(stopwords.words('english'))
# Lowercase each document, split it by white space and filter out stopwords
texts = [[word for word in document.lower().split() if word not in stoplist and stop]
         for document in raw_corpus]

# Count word frequencies
from collections import defaultdict
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

# Only keep words that appear more than once
processed_corpus = [[token for token in text if frequency[token] > 1] for text in texts]
dictionary = corpora.Dictionary(processed_corpus)
print(dictionary)
bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus]

#Code for LdaMulticore taken from
lda_model = LdaMulticore(corpus=bow_corpus,
                         id2word=dictionary,
                         random_state=100,
                         num_topics=100,
                         passes=10,
                         chunksize=1000,
                         batch=False,
                         alpha='asymmetric',
                         decay=0.5,
                         offset=64,
                         eta=None,
                         eval_every=0,
                         iterations=100,
                         gamma_threshold=0.001,
                         per_word_topics=True)

# save the model
lda_model.save('lda_model.model')

# See the topics
# lda_model.print_topics(-1)

def here_are_the_topics():
    for i in range(0, lda_model.num_topics-1):
       print(lda_model.print_topic(i))

here_are_the_topics()

print("--- %s seconds ---" % (time.time() - start_time))


print("Total number of tweets taken into account is:", len(processed_corpus))
# print()


# ------------------------------------------------------------------------------------------------------------------
