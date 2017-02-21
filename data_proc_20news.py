# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 09:38:22 2017

@author: ekhongl
"""

import os
import numpy as np
import pandas as pd
import re

import nltk
#from nltk.tokenize import PunktSentenceTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
ps = PorterStemmer()
lemma = WordNetLemmatizer()

from sklearn.feature_extraction.text import CountVectorizer


import gensim
from gensim import corpora, models


dir_wd = os.getcwd()

dir_src = os.path.join(dir_wd, 'data/raw_20news/20news-18828')

dir_src_classes = list( map(lambda x: os.path.join(dir_src, x ), os.listdir(dir_src)) )


dat = []
dat_y = []
dat_y_cat = []

for i in range(0,len(dir_src_classes)):
    
    print('Currently loading the following topic (iteration ' + str(i) + '):\n \t' + dir_src_classes[i])
    dir_src_classes_file = list( map(lambda x: os.path.join(dir_src_classes[i], x), os.listdir(dir_src_classes[i])) )
    
    for ii in range(0, len(dir_src_classes_file)):
        
        dat_y.append(i)
        
        with open(dir_src_classes_file[ii], 'r') as file:
            dat.append(file.read().replace('\n', ' '))

#export data
pd.DataFrame( { 'labels' : dat, 
                'documents' : dat_y}). \
                to_csv(os.path.join(dir_wd,'data/raw_20news/20news.csv'),
                    index=False)

print('------- Data cleaning -------')                
stopwords_en = stopwords.words('english')
dat_clean = []
for i in range(len(dat)):

    ''' tokenization and punctuation removal '''
    # uses nltk tokenization - e.g. shouldn't = [should, n't] instead of [shouldn, 't]
    tmp_doc = nltk.tokenize.word_tokenize(dat[i].lower())
    
    # split words sperated by fullstops
    tmp_doc_split = [w.split('.') for w in tmp_doc if len(w.split('.')) > 1]
    # flatten list
    tmp_doc_split = [i_sublist for i_list in tmp_doc_split for i_sublist in i_list]
    # clean split words
    tmp_doc_split = [w for w in tmp_doc_split if re.search('^[a-z]+$',w)]
    
    # drop punctuations
    tmp_doc_clean = [w for w in tmp_doc if re.search('^[a-z]+$',w)]
    tmp_doc_clean.extend(tmp_doc_split)

    ''' stop word removal'''
    tmp_doc_clean_stop = [w for w in tmp_doc_clean if w not in stopwords_en]
    #retain only words with 2 characters or more
    tmp_doc_clean_stop = [w for w in  tmp_doc_clean_stop if len(w) >2]
    
    ''' stemming (using the Porter's algorithm)'''
    tmp_doc_clean_stop_stemmed = [ps.stem(w) for w in  tmp_doc_clean_stop]
    dat_clean.append(tmp_doc_clean_stop_stemmed)
    
    #print progress
    if i % 100 == 0: print( 'Current progress: ' + str(i) + '/' + str(len(dat)) )

#save cleaned data
pd.DataFrame( { 'document' : dat_clean, 
                'label' : dat_y}). \
                to_csv(os.path.join(dir_wd,'data/clean_20news.csv'),
                    index=False)



# all_words = [word for doc in dat_clean for word in doc]
# pd_all_words = pd.DataFrame({'words' : all_words})
# pd_unq_word_counts = pd.DataFrame({'count' : pd_all_words.groupby('words').size()}).reset_index().sort('count', ascending = False)
# pd_unq_word_counts_filtered = pd_unq_word_counts.loc[pd_unq_word_counts['count'] >= 100]
# list_unq_word_filtered = list( pd_unq_word_counts_filtered.ix[:,0] )


# vec = CountVectorizer(input = 'content', lowercase = False, vocabulary = list_unq_word_filtered)


# iters = list(range(0,len(dat_clean),500))
# iters.append(len(dat_clean))
# dtm = []
# for i in range(len(iters)-1):
    # dtm = np.concatenate((dtm, list(map(lambda x: vec.fit_transform(x).toarray().astype(np.int32), dat_clean[iters[i]:iters[i+1]] )) ), axis = 0)
    # print(str(i))


# data = vec.fit_transform(tmp).toarray()
# print(data)

# import numpy as np
# list(map(lambda x: vec.fit_transform(x).toarray(), tmp )).shape

# [i for i in map(lambda x: vec.fit_transform(x).toarray(), tmp )]





# dictionary = corpora.Dictionary(dat_clean)
# dictionary.filter_extremes(no_below=50, no_above=0.7, keep_n=100000)


# dtm = [dictionary.doc2bow(doc) for doc in dat_clean]



# tmp = [list(map(lambda x : doc.count(x), list_unq_word_filtered)) for doc in dat_clean]


# tmp = list(map(lambda y: list( map( lambda x : y.count(x), list_unq_word_filtered ) ), dat_clean))


# [doc.count(w_dic) for doc in dat_clean for w_dic in list_unq_word_filtered]

















import pandas as pd
import matplotlib.pyplot as plt
plt.hist(pd_all_words['words'].value_counts()[300:3000])
plt.title("Plot of word counts")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()


for doc in dat_clean if 








pd_unq_word_counts_filtered.ix[:,0]


count_vect = CountVectorizer(list(pd_unq_word_counts_filtered.ix[:,0]))
count_vect.fit_transform(dat_clean[0]).shape
    
all_words = nltk.FreqDist(unq_words)


set(dat_clean[0])
















dictionary = corpora.Dictionary(dat_clean)
    
# convert tokenized documents into a document-term matrix
dtm = [dictionary.doc2bow(doc) for doc in dat_clean]

# generate LDA model
ldamodel = gensim.models.ldamulticore.LdaMulticore( workers=3,
                                           corpus = dtm,
                                           id2word = dictionary, 
                                           num_topics=20, 
                                           passes=100, 
                                           random_state=123)









# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()
texts = []
bcfs = []
tokenizer = RegexpTokenizer(r'\w+')
for i in raw_data:
    
    # clean and tokenize document string
    #raw = str(raw_data[i].lower())    
    raw = raw_data[i]
    raw_str = pd.Series.to_string(raw)
    raw_str = raw_str.lower()
    tokens = raw_str.split()
    
    #StanfordTokenizer().tokenize(raw_str)
    #tokens = tokenizer.tokenize(raw_str.split())

    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in en_stop]
    removedP_tokens = [w for w in stopped_tokens if re.search('^[a-z]+$',w)]
    cleaned_tokens = [w for w in removedP_tokens if len(w) >= 3]
    
    # stem tokens
    #stemmed_tokens = [p_stemmer.stem(i) for i in cleaned_tokens]
    normalized = [lemma.lemmatize(i) for i in cleaned_tokens]
    normalized_pos = pos_tag(normalized)
    print(normalized_pos)
    
    # add tokens to list
    texts.append(normalized)







#------------------------------------------------------------------------------  
alt            0
comp           1,2,3,4,5
misc            6,
rec            7,8,9,10,
sci            11,12,13,14
soc            15,
talk            16,17,18,19































