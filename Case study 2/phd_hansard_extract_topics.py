# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 11:45:03 2015

this script imports lda topic results for each year, and creates two inputs:
- a dataframe containing topwords and their probabilities, 
- a dataframe with topic probabilities distributions for each speaker

@author: S
"""

import os
import pandas as pd
from gensim import models, corpora

###############################################################################
#set parameters
myPath = "C:\\Users\\S\\Documents\\Uni\\PhD\\final_thesis\\data\\"
num_topics = 20
num_topwords = 600
###############################################################################
#obtain list of lda files, corpora, indices for each time slice
files_lda = list()
for file in os.listdir(myPath + "hansard_results_lda\\"):
    if file.endswith(".lda"):
        files_lda.append(file)

files_corpus = list()
for file in os.listdir(myPath + "hansard_results_lda\\"):
    if file.endswith(".mm"):
        files_corpus.append(file)

files_index = list()
for file in os.listdir(myPath + "hansard_input_lda\\"):
    if "index" in file:
        files_index.append(file)
###############################################################################
#create two dataframes: one having topwords and probabilities, the other 
#prevalences per speaker         
for f in range(0,len(files_lda)):
    lda = models.ldamodel.LdaModel.load(myPath + "hansard_results_lda\\" + files_lda[f])
    corpus = corpora.MmCorpus(myPath + "hansard_results_lda\\" + files_corpus[f])
    
    
    ###########################################################################
    #dataframe for topics and topwords
    index = range(0,num_topwords)
    columns = [str(i) + "_topwords" for i in range(0,num_topics)] + [str(i) + 
              "_probs" for i in range(0,num_topics)]
    
    topics = pd.DataFrame(index=index, columns=columns)
    
    for i in range(0,num_topics):
        label = str(i)+ "_topwords"
        topics[label] = pd.Series([p[1] for p in lda.show_topic(i,num_topwords)])
        label = str(i)+ "_probs"
        topics[label] = pd.Series([p[0] for p in lda.show_topic(i,num_topwords)])    
    
    #export    
    topics.to_csv(myPath + "hansard_results_lda\\" + files_lda[f][:9] + "_topics.csv", 
                  encoding = 'utf-8')
    
    ###########################################################################
    #dataframe with topic probability distribution for each speeech
    distr = pd.read_csv(myPath + "hansard_input_lda\\" + files_index[f], 
                        names = ["old_index", "sessionID", "speakerID", "date"])    
    
    #create columns for each topic
    col_topic = [str(i) + "_topic" for i in range(0,num_topics)]
    for c in col_topic:
        distr[c] = 0
    
    #fill the values with the topic probabilities
    for d in range(0,len(corpus)):
        t_probs = lda[corpus[d]]
        for p in t_probs:
            distr.loc[d, str(p[0]) +"_topic"] = p[1]
    #esport
    distr.to_csv(myPath + "hansard_results_lda\\" + files_lda[f][:9] + "_topic_distr.csv")
    
    