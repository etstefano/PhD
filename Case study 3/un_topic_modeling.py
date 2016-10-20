# -*- coding: utf-8 -*-
"""
Created on Tue May 05 11:08:38 2015

@author: S
"""

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from gensim import corpora, models
import os
import nltk

###############################################################################
myPath = "C:\\Users\\S\\Documents\\Uni\\UNspeeches\\"

with open(myPath + "input_clean\\un_all_speeches.txt") as f:
    speeches = f.read()
    
speeches = [[word for word in document.split()]
        for document in speeches.split("\n")]
del speeches[-1]

dictionary = corpora.Dictionary(speeches)
dictionary.save(myPath + "un_dict.dict") # store the dictionary, for future reference
print(dictionary)

corpus = [dictionary.doc2bow(text) for text in speeches]
corpora.MmCorpus.serialize(myPath + "un_corpus.dict", corpus)

with open(myPath + "input_clean\\speech_per_year.txt") as f:
    f = f.read()
    timeslices = [n for n in f.split("\n")]
    del timeslices[-1]
    timeslices = [int(n) for n in timeslices]
    


###############################

corpus = corpora.MmCorpus(myPath + "un_corpus.dict")
dictionary = corpora.Dictionary.load(myPath + "un_dict.dict")

model = gensim.models.DtmModel(myPath + "dtm-master\\bin\\dtm-win64.exe", corpus, timeslices, 
                               num_topics=20, id2word=dictionary)
                            
model = models.ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=20)
model.save(myPath + "un_model.lda")

###############################
#UN LDA per year

years = range(1970, 2015)
myPath = "C:\\Users\\S\\Documents\\Uni\\UNspeeches\\"
files = list()
for file in os.listdir(myPath + "input_clean\\"):
    if file.endswith(".txt"):
        files.append(file)


for y in years: 
    sublist = [speech for speech in files if str(y) in speech]
    speeches = list()
    
    for text in sublist: 
        with open(myPath + "\\input_clean\\" + text) as t:
            speeches.append([word for word in t.read().split(" ")])
               # if word not in polstop])
    
    frq = nltk.FreqDist([item for sublist in speeches for item in sublist])
    polstops = [w[0] for w in frq.most_common(int(0.005*len(frq)))]
    
    speeches = [[word for word in s if word not in polstops] for s in speeches]    
    
    dictionary = corpora.Dictionary(speeches)
    dictionary.save(myPath + "input_lda\\" + str(y) + "_un_dict.dict") 
    
    corpus = [dictionary.doc2bow(text) for text in speeches]
    corpora.MmCorpus.serialize(myPath + "input_lda\\" + str(y) + "_un_corpus.mm", corpus)
    
    lda = models.ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=8, passes = 10,
        iterations=500)
        
    lda.save(myPath + "input_lda\\" + str(y) + "_lda_results.lda")



