# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 10:20:23 2015

this script takes as input the speeches made each year, and perform LDA topic 
modeling. We store the results in the folder hansard_results

@author: S
"""

#import the libraries needed
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import random
import os
from gensim import corpora, models

random.seed(89)

myPath = "C:\\Users\\S\\Documents\\Uni\\PhD\\final_thesis\\data\\"

###############################################################################
#obtain list of files with input and speakerID sequence
files_speeches = list()
for file in os.listdir(myPath + "hansard_input_lda\\"):
    if "all_speeches" in file:
        files_speeches.append(file)

files_id_sequence = list()
for file in os.listdir(myPath + "hansard_input_lda\\"):
    if "seq" in file:
        files_id_sequence.append(file)
###############################################################################
#for each time window, perforf lda
for t in range(0,len(files_speeches)):
    
    #get the speeches
    with open(myPath + 'hansard_input_lda\\' + files_speeches[t]) as f:
        speeches = f.read()
        speeches = [[word for word in document.split()]
            for document in speeches.split("\n")]
    del speeches[-1] #last line is empty
    
    #get the speaker IDs
#    with open(myPath + 'hansard_input_lda\\' + files_id_sequence[t]) as f:
#        speakerID = [int(m.replace('\n', '')) for m in f.readlines()]

    #create dictionary  
    print "creating dictionary for years " + files_speeches[t][:9]  
    dictionary = corpora.Dictionary(speeches)
    #dictionary.filter_extremes(no_below = 5)
    dictionary.save(myPath + "hansard_results\\" + files_speeches[t][:9] + 
        "_hansard_dict.dict") 
    
    #create corpus
    corpus = [dictionary.doc2bow(text) for text in speeches]
    corpora.MmCorpus.serialize(myPath + "hansard_results\\" + files_speeches[t][:9]
        + "_hansard_corpus.mm", corpus)
    
    print "performing lda on years " + files_speeches[t][:9]
    lda = models.ldamodel.LdaModel(corpus, id2word=dictionary, num_topics= 20, passes = 10,
        iterations=500)
        
    lda.save(myPath + "hansard_results\\" + files_speeches[t][:9] + 
        "_hansard_lda_results.lda")
    
        
    
    
        