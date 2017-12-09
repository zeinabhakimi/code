#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 15:23:13 2017

@author: zeinabhakimi
"""


import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob


# Load the data
train = pd.read_csv('train.tsv', sep='\t')
test = pd.read_csv('test.tsv', sep='\t')
train.shape


def split_into_lemmas(message):
  
    output_message = str(message).lower()
  
        
    words = TextBlob(output_message).words
    # for each word, take its "base form" = lemma 
    return [word.lemma for word in words]


train_lemma_phrase = []
for phrase in train.Phrase:
    train_lemma_phrase.append(' '.join(split_into_lemmas(phrase)))

test_lemma_phrase = []
for phrase in test.Phrase:
    test_lemma_phrase.append(' '.join(split_into_lemmas(phrase)))
    


train_output = pd.DataFrame({
    'PhraseId': train.PhraseId,
    'Phrase':  train_lemma_phrase,
    'Sentiment': train.Sentiment
})



test_output = pd.DataFrame({
    'PhraseId': test.PhraseId,
    'Phrase':  test_lemma_phrase
})
train_output.to_csv("result_train.csv", index = False, encoding='ascii')
test_output.to_csv("result_test.csv", index = False, encoding='ascii')
