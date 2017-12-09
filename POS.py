#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 15:17:10 2017

@author: zeinabhakimi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import lil_matrix
from sklearn.svm import SVC
train = pd.read_csv('result_train.csv')
test = pd.read_csv('result_test.csv')

train.fillna(" ", inplace = True)
test.fillna(" ", inplace = True)


# train = train[:10]
# test = test[:10]

split_indexs = int(len(train)/2)
train_test = train[: split_indexs]
train_train = train[split_indexs:]
columns = ['PhraseId', 'Phrase', 'Sentiment']
train_test= pd.DataFrame(train_test, columns=columns)
train_train= pd.DataFrame(train_train, columns=columns)


train = np.array(train)
test = np.array(test)

# train_sentiment = train[:,-1]
train = train[:, :2]

print (train.shape)
print (test.shape)

data = np.concatenate((train, test), axis=0)
print (data.shape)
columns = ['Phrase', 'PhraseId']
data= pd.DataFrame(data, columns=columns)
test = pd.DataFrame(test, columns=columns)

from textblob import TextBlob
def extract_verbs_and_adj(phrase):
    words = []
    
    tags = TextBlob(phrase).tags
    for tag in tags:
        if tag[1][:2] == "VB" or tag[1][:2] == "JJ":
            words.append(tag[0])
    return words

def looping_extract_verbs_and_adj(data_set):
    phrases_vb_adj = []

    for (index, phrase) in enumerate(data_set):
        
        phrases_vb_adj.append(' '.join(extract_verbs_and_adj(phrase)))
    return phrases_vb_adj

data_vb_adj = looping_extract_verbs_and_adj(data.Phrase)

v1 = CountVectorizer(ngram_range=(0, 4),  min_df=2, max_df=0.95, max_features=1000)
v1.fit(data_vb_adj)


#train_test_vb_adj = looping_extract_verbs_and_adj(train_test.Phrase)
train_train_vb_adj = looping_extract_verbs_and_adj(train_train.Phrase)
test_vb_adj = looping_extract_verbs_and_adj(test.Phrase)

def add_vb_adj_sentiment(phrases_vb_adj):
    X2 = v1.transform(phrases_vb_adj)

    sentiments = []
    for feature in v1.get_feature_names():
        sentiment = TextBlob(feature).sentiment
        value = sentiment.polarity * sentiment.subjectivity
        sentiments.append(value)
   

    from scipy.sparse import csr_matrix

    X3 = csr_matrix(X2.shape)
    for index in range(X2.shape[0]):
        row = X2[index]
        for col in range(row.shape[1]):
            X3[index, col] = X2[index, col] * sentiments[col]

    return X3
    #return X3
#X_train_test = add_vb_adj_sentiment(train_test_vb_adj)
X_train_train = add_vb_adj_sentiment(train_train_vb_adj)
X_test = add_vb_adj_sentiment(test_vb_adj)

clf = SVC()
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)

# y = np.array([1,2,3,4,5])
#clf.fit(train_train_vb_adj, train_train.Sentiment) 
clf.fit(X_train_train, train_train.Sentiment)
print ("predicting...")
#y_pred = clf.predict(test_vb_adj)
y_test_predict = clf.predict(X_test)
results0_test = pd.DataFrame({
    'PhraseId': test.PhraseId,
    'Sentiment': y_test_predict
})
results0_test.to_csv('results2_test.csv', index=False)
print ("done.")
