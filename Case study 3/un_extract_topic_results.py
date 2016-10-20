# -*- coding: utf-8 -*-
"""
Created on Mon May 11 17:03:07 2015

@author: S
"""

import os
import pandas as pd
from gensim import corpora, models

myPath = "C:\\Users\\S\\Documents\\Uni\\UNspeeches\\"

files = list()
for file in os.listdir(myPath + "input_lda\\"):
    if file.endswith(".lda"):
        files.append(file)

for f in files:
    lda = models.ldamodel.LdaModel.load(myPath + "input_lda\\" + f)
    corpus = corpora.MmCorpus(myPath + "input_lda\\" + f[:4] + "_un_corpus.mm")
    
    #dataframe for topics and topword
    index = range(0,1000)
    columns = [str(i) + "_topwords" for i in range(0,8)] + [str(i) + "_probs" for i in range(0,8)]
    
    topics = pd.DataFrame(index=index, columns=columns)
    
    for i in range(0,8):
        #p = pd.Series([a[0] for a in lda.show_topic(0,1000)])
        label = str(i)+ "_topwords"
        topics[label] = pd.Series([p[1] for p in lda.show_topic(i,1000)])
        label = str(i)+ "_probs"
        topics[label] = pd.Series([p[0] for p in lda.show_topic(i,1000)])    
        
    topics.to_csv(myPath + "output_lda\\" + f[:4] + "_topics.csv", encoding = 'utf-8')
    
    #dataframe for doc and topics
    index = range(0,8)
    columns = [str(i)+ "_doc" for i in range(0, len(corpus))]
    
    docs = pd.DataFrame(index=index, columns=columns)
    
    for i in range(0, len(corpus)):
        t_prevalences = lda[corpus[i]]
        for t in t_prevalences:
            docs.loc[t[0],str(i)+"_doc"] = t[1]
                
    docs = docs.fillna(value = 0)
    
    docs.to_csv(myPath + "output_lda\\" + f[:4] + "_docs.csv")