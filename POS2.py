#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 19:18:00 2017

@author: zeinabhakimi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from sklearn.svm import SVC
from textblob.classifiers import NaiveBayesClassifier
# Load the data
train = pd.read_csv('result_train.csv', encoding='ascii')
test = pd.read_csv('result_test.csv', encoding='ascii')
train.fillna('', inplace=True)
test.fillna('', inplace=True)

def text_blob_sentiment(phrase):
    try:
        return TextBlob(phrase).sentiment.polarity
    except:
        return 0

def get_text_blob_sentiments(phrases):
    sentiments = map(text_blob_sentiment, phrases)
    return pd.DataFrame({'sentiment': list(sentiments)})

split_index = int(len(train) / 2)
cv = train[:split_index]
train = train[split_index:]



X_train_sent = get_text_blob_sentiments(train.Phrase)
X_cv_sent = get_text_blob_sentiments(cv.Phrase)
X_test_sent = get_text_blob_sentiments(test.Phrase)

svc = SVC()

print ("training SVC...")
svc.fit(X_train_sent, train.Sentiment)

# Predict using cross validation data
#cv_pred = svc.predict(X_cv_sent)
#results_cv = pd.DataFrame({
#    'PhraseId': cv.PhraseId,
#   'Predicted': cv_pred,
#   'Sentiment': cv.Sentiment
#})
#results_cv.to_csv('results_train.csv', index=False)

print ("predicting...")
y_pred = svc.predict(X_test_sent)

results_test = pd.DataFrame({
    'PhraseId': test.PhraseId,
    'Sentiment': y_pred
})

results_test.to_csv('results_test.csv', index=False)
print ("done.")