# -*- coding: utf-8 -*-


import os
import numpy as np
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer



os.chdir('/home/ekhongl/Codes/DL - Topic Modelling')

dat = pd.read_csv('data/clean_20news.csv', sep=",")


docs = [ast.literal_eval(doc) for doc in dat['document'].tolist()]

all_words = [word for doc in docs for word in doc]
pd_all_words = pd.DataFrame({'words' : all_words})
pd_unq_word_counts = pd.DataFrame({'count' : pd_all_words.groupby('words').size()}).reset_index().sort('count', ascending = False)
pd_unq_word_counts_filtered = pd_unq_word_counts.loc[pd_unq_word_counts['count'] >= 150]
list_unq_word_filtered = list( pd_unq_word_counts_filtered.ix[:,0] )
len(list_unq_word_filtered)


vec = CountVectorizer(input = 'content', lowercase = False, vocabulary = list_unq_word_filtered)

iters = list(range(0,len(docs),500))
iters.append(len(docs))
dtm = np.array([] ).reshape(0,len(list_unq_word_filtered))
for i in range(len(iters)-1):
    dtm = np.concatenate( (dtm, list(map(lambda x: vec.fit_transform(x).toarray().sum(axis=0), docs[iters[i]:iters[i+1]] )) ), axis = 0)
    print(str(i))

colnames = list_unq_word_filtered
colnames.insert(0,'_label_')
                
pd.DataFrame(data = np.c_[dat['label'].values, dtm], 
             columns = colnames). \
             to_csv( 'data/dtm_20news.csv', index = False)
