# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 17:46:47 2016

@author: S
"""

import sys  
reload(sys)  
sys.setdefaultencoding('utf8')

import os 
import numpy as np
import pandas as pd

from gensim.models import doc2vec
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

myPath = "C:\\Users\\S\\Documents\\Uni\\UNspeeches\\"

files = list()
for file in os.listdir(myPath + "input_clean2\\"):
    if file.endswith(".txt"):
        files.append(file)
        
#load doc2vec model
model = doc2vec.Doc2Vec.load(myPath + "doc2vec\\" + 'un_speeches.doc2vec')  

#for each year, create two files:
#   1 list of index for each country
#   2 adj matrix with cosine similarity
#then write them down for later use

years_range = range(1962, 2015)

for yy in years_range:
    print yy
    #initialize country list 
    countries = []
    for cc in model.vocab.keys():
        try:
            if str(yy) in cc:
                countries.append(cc)
        except TypeError:
            continue
    #initialize adj mat
    adj = np.zeros([len(countries), len(countries)])
    #for each couple of countries, compute cosine similarity
    for ii in range(0,len(countries)):
        for jj in range(ii+1, len(countries)):
            adj[ii,jj] = model.n_similarity([countries[ii]], [countries[jj]])
    
    #export adj matrix
    np.savetxt(myPath + "doc2vec\\" + str(yy) + "_adj.csv", adj,
               fmt='%.9f',)
               
    #export country list/indices
    country_series = pd.Series(countries)
    country_series.to_csv(myPath + "doc2vec\\" + str(yy) + "_countries.csv")
    
        
        