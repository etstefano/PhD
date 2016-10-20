# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 08:38:43 2015
This script creats time series from the topic results, and calculate their
transfer entropies

@author: S
"""

import sys  
reload(sys)  
sys.setdefaultencoding('utf8')

import os
import pandas as pd
from sklearn import decomposition

import random
random.seed(8)

###############################################################################
#set parameters
myPath = "C:\\Users\\S\\Documents\\Uni\\PhD\\final_thesis\\data\\"
nTopics = 20
nShuffling = 1000 #number of times you shuffle a time series, needed for testing significance
###############################################################################
###############################################################################
def f_create_time_mat(distr,col_topics):
    
    time_mat = pd.DataFrame(index = sorted(distr.speakerID.unique()), 
                            columns = sorted(distr.date.unique()))    
    
    grouped = distr.groupby(['speakerID', 'date'])
    for name, group in grouped: 
        if len(group) > 1:
            time_mat.loc[name[0],name[1]] = group['pca'].mean(axis = 0)
        else:
            time_mat.loc[name[0],name[1]] = float(group['pca'].values)
    
    return time_mat

###############################################################################
###############################################################################
#get list of files
files_distr = list()
for file in os.listdir(myPath + "hansard_results_lda\\"):
    if "topic_distr" in file:
        files_distr.append(file)
###############################################################################
#cycle through each time period 
for y in range(0,len(files_distr)):
    
    print "working on years " + files_distr[y][:9]
    
    #import dataframe
    distr = pd.read_csv(myPath + "hansard_results_lda\\" + files_distr[y], 
                        index_col = 0)
    distr.date = pd.to_datetime(distr.date)

    #get topic column names
    col_topics = distr.columns[4:]
    
    #normalise topic probs
    distr[col_topics] = distr[col_topics].div(distr[col_topics]
                                        .sum(axis=1), axis=0)
    
    #obtain pca for each speech
    pca_model = decomposition.RandomizedPCA(n_components = 1)
    distr['pca'] = pca_model.fit_transform(distr[col_topics].values)

    
    #get dates 
    dates = sorted(distr.date.unique())
    descriptive_df = pd.DataFrame(index = range(0, len(dates)), 
                                  columns = ['date', 'nSpeakers'])
    for i in range(0,len(dates)):
        descriptive_df.loc[i, 'date'] = dates[i]
        descriptive_df.loc[i, 'nSpeakers'] = len(distr.speakerID[distr.date == 
                                                 dates[i]].unique())
    descriptive_df.to_csv(myPath + "hansard_te\\" + files_distr[y][:9] + "_des.csv")
    
    #create time series matrix
    time_mat = f_create_time_mat(distr,col_topics)
    time_mat.to_csv(myPath + "hansard_te\\" + files_distr[y][:9] + "_timeseries.csv")


