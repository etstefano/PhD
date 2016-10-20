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
from jpype import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
def f_average_samedate(distr,dates,speakers,col_topics):
    for s in speakers:
        for d in dates: 
            if len(distr[(distr.speakerID == s) & (distr.date == d)]) > 1:
                print "averaging for speaker " + str(s)
                mini_df = distr[(distr.speakerID == s) & (distr.date == d)]
                avg = mini_df.drop_duplicates(subset = 'date')
                avg_series = mini_df[col_topics].mean(axis = 0).transpose()
                for c in range(0,len(col_topics)):
                    avg[col_topics[c]] = avg_series[c] 
                distr.loc[mini_df.index] = np.nan
                distr.dropna()
                #distr = distr.drop(distr.loc[mini_df.index])
                distr = pd.concat([distr,avg])

    return distr                
                
###############################################################################
###############################################################################
def f_get_timeseries(s,distr,col_topics):
    """
    function creating matrix of timeseries for agent s
    """
    #get distr for speaker and dates
    distr_s = distr[distr.speakerID == s]
    dates_s = distr_s.date.unique().astype("datetime64[D]")
    dates_s_minus =  [x-1 for x in dates_s]
    
    
    #get distr for speakers who speak a day before 
    distr_temp = distr[distr.date.isin(dates_s_minus)]
    distr_temp = distr_temp[distr_temp.speakerID != s]
    speakers_infl = distr_temp.speakerID.unique()    
    
    #do pca for entire data
    distr_conc = pd.concat([distr_s, distr_temp])   
    pca_model = decomposition.RandomizedPCA(n_components = 1)
    distr_conc['pca'] = pca_model.fit_transform(distr_conc[col_topics].values)
    
    #create and fill timeseries    
    ts_mat = pd.DataFrame(index = sorted(dates_s), columns = np.append(s,speakers_infl))
    for d in ts_mat.index.values.astype("datetime64[D]"):
        for sp in distr_conc.speakerID:
            try: 
                if sp == s:
                    ts_mat.loc[d,sp] = float(distr_conc.pca[(distr_conc.speakerID == s) &
                                        (distr_conc.date == d)].values)
                else:
                    ts_mat.loc[d,sp] = float(distr_conc.pca[(distr_conc.speakerID == s) &
                                        (distr_conc.date == d-1)].values)
            except TypeError: 
                continue
        
    return ts_mat

###############################################################################
###############################################################################
#get list of files
files_distr = list()
for file in os.listdir(myPath + "hansard_results_lda\\"):
    if "topic_distr" in file:
        files_distr.append(file)
###############################################################################
#cycle through each time period 
for y in range(0,len(files_distr))[:1]:
    
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
    
    #get dates 
    dates = distr.date.unique()
    speakers = distr.speakerID.unique()
    nSpeakersDay = [len(distr[distr.date == i]) for i in dates]
    
    #average out for speeches made in the same day
    distr = f_average_samedate(distr,dates,speakers,col_topics)
    distr = distr.dropna()
    distr.to_csv(myPath + "hansard_results_lda\\" + files_distr[y][:9] +
                '_topic_distr_avg.csv')
    
    
    #for each speaker, get rel
    for s in speakers[:1]: 
        print s
        ts_mat = f_get_timeseries(s, distr, col_topics)

