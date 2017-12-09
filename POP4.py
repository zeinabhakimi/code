#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 07:37:11 2017

@author: zeinabhakimi
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 07:36:21 2017

@author: zeinabhakimi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import lil_matrix
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from textblob import TextBlob
'''
from Athar import ExtractText
from Athar import ReplaceText
from Athar import MapToSynsets

'''
train = pd.read_csv('result_train.csv')
test = pd.read_csv('result_test.csv')

train.fillna(" ", inplace = True)
test.fillna(" ", inplace = True)


# train = train[:10]
# test = test[:10]

split_indexs = int(len(train)/4)
train_test = train[: split_indexs]
#train_train = train[split_indexs:]
columns = ['PhraseId', 'Phrase', 'Sentiment']
#train_test= pd.DataFrame(train_test, columns=columns)
train_train= pd.DataFrame(train, columns=columns)


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


def scikit_learn(train_set, train_labels):
    text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1,2),  min_df=2, max_df=0.95, max_features=1000)),
                     ('tfidf', TfidfTransformer()),
                     ('clf', OneVsOneClassifier(LinearSVC())),
                     ])
    X_train = np.asarray(train_set)
    text_clf = text_clf.fit(X_train, np.asarray(train_labels))
    return text_clf

def extract_verbs_and_adj(phrase):
    words = []
    
    tags = TextBlob(phrase).tags

    for tag in tags:
        if tag[1][:2] == "VB" or tag[1][:2] == "JJ":
           # print(tag[1][:2])
            words.append(tag[0])
   # print (words)
    return words

def looping_extract_verbs_and_adj(data_set):
    phrases_vb_adj = []

    for (index, phrase) in enumerate(data_set):
        
        phrases_vb_adj.append(' '.join(extract_verbs_and_adj(phrase)))
    return phrases_vb_adj

data_vb_adj = looping_extract_verbs_and_adj(data.Phrase)
#print (data_vb_adj)

#rain_test_vb_adj =  looping_extract_verbs_and_adj(train_test.Phrase)
train_train_vb_adj = looping_extract_verbs_and_adj(train_train.Phrase)
test_vb_adj = looping_extract_verbs_and_adj(test.Phrase)

for row in train_train_vb_adj:
    #print(row)
    for tokens in row:
        bigrams = [tokens[i:i+2] for i,w in enumerate(tokens) if i+2 <= len(tokens)]
        print(bigrams)
    for x in bigrams:
        print(train_f.append(x))
print (train_f) 

#ext_clf = scikit_learn(train_train_vb_adj, train_train.Sentiment)


print ("predicting...")
'''
predicted = text_clf.predict(np.asarray(train_train_vb_adj))

y_pred = clf.predict(test_vb_adj)

results0_test = pd.DataFrame({
    'PhraseId':  train_train.PhraseId,
    'Sentiment': predicted
})
results0_test.to_csv('results0_train.csv', index=False)
print ("done.")
'''



