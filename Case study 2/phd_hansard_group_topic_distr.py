# -*- coding: utf-8 -*-
"""
Created on Tue Nov 03 13:09:13 2015

This script imports topic distributions per speech, and groups them by person 
and by party for each timeslie

@author: S
"""


import sys  
reload(sys)  
sys.setdefaultencoding('utf8')

import os
import pandas as pd

###############################################################################
#set parameters
myPath = "C:\\Users\\S\\Documents\\Uni\\PhD\\final_thesis\\data\\"

###############################################################################
#get list of files
files_distr = list()
for file in os.listdir(myPath + "hansard_results_lda\\"):
    if "topic_distr" in file:
        files_distr.append(file)

files_speeches = list()
for file in os.listdir(myPath + "hansard_input_lda\\"):
    if "all_speeches" in file:
        files_speeches.append(file)

#get hansard info
info = pd.read_csv(myPath + "hansard_raw\\Hansard_speeches_1935-2014_member_info.csv", 
                   names = ["sessionID", "memberID", "speakerID", "speakerName",
                            "firstname","lastname", "constituency", "party"])
###############################################################################
###############################################################################
#functions
###############################################################################
###############################################################################
def func_speaker_probs(distr_temp, col_topics):
#creates weighted probability vectors for each speaker
 
    #sum of words said by the speaker
    sum_words = distr_temp.nwords.sum()
    
    #normalise the row probability vectors of the original distribution
    distr_temp[col_topics] = distr_temp[col_topics].div(distr_temp[col_topics]
                                        .sum(axis=1), axis=0)
    
    #multiply each probability with nwords, then divide by nwords
    speaker_prob_vec = distr_temp[col_topics].apply(lambda x: 
                              (x*distr_temp.nwords)).sum(axis = 
                              0).div(sum_words).tolist()
    
    return speaker_prob_vec
###############################################################################
def func_party_probs(distr_temp_party, col_topics):
#for each party calculate the weigthed probability
    
    #sum of words for each party
    sum_words = distr_temp_party.nwords.sum()
    
    #normalise the row probability vectors of the original distribution
    distr_temp_party[col_topics] = distr_temp_party[col_topics].div(distr_temp_party[col_topics]
                                        .sum(axis=1), axis=0)
    
    #ultiply each probability with nwords, then divide by nwords
    party_prob_vec = distr_temp_party[col_topics].apply(lambda x: 
                              (x*distr_temp_party.nwords)).sum(axis = 
                              0).div(sum_words).tolist()
                              
    return party_prob_vec

###############################################################################
###############################################################################


    
#for each timeslice, group by speaker ID and sum probabilities
for y in range(0,len(files_distr))[1:]:
    
    print "working on years " + files_distr[y][:9]
    
    #import dataframe
    distr = pd.read_csv(myPath + "hansard_results_lda\\" + files_distr[y])
    col_topics = [t for t in distr.columns if "_topic" in t]    
    
    #import speeches and get number of words
    with open(myPath + 'hansard_input_lda\\' + files_speeches[y]) as f:
        speeches = f.read()
        speeches = [[word for word in document.split()]
            for document in speeches.split("\n")]
    del speeches[-1] #last line is empty
    distr['nwords'] = [len(x) for x in speeches]
    
    #slice info dataframe according to the session and ID present
    info_sliced = info[info.sessionID.isin(distr.sessionID.unique())]
    ####IMPORTANT: speakerID in distr is memberID in info!! --> correct error
    info_sliced = info_sliced[info_sliced.memberID.isin(distr.speakerID.unique())]
    #delete session columns and drop duplicates
    del info_sliced['sessionID']
    info_sliced = info_sliced.drop_duplicates(subset = 'memberID')
    info_sliced = pd.concat([info_sliced,pd.DataFrame(columns=col_topics)])

    ###########################################################################
    #create dataset by speaker
    print "creating dataset by speaker"
    for m in info_sliced.memberID:
        distr_temp = distr[distr.speakerID == m]
        ind = info_sliced[info_sliced.memberID == m].index  

        info_sliced.loc[ind, col_topics] = func_speaker_probs(distr_temp,col_topics)
    
    #export dataframe
    info_sliced.to_csv(myPath + "hansard_results_lda\\" + files_distr[y][:9] +
                        "_top_distr_speaker.csv")
    
    ###########################################################################
    ###########################################################################
    #create dataset by party, from 2000-01 onwards
    if y != 0: 
        
        print "creating dataset by party"
        parties = info_sliced.party.unique()
        party_distr = pd.DataFrame(index = range(0,len(parties)))
        party_distr['party'] = parties
        party_distr = pd.concat([party_distr,pd.DataFrame(columns=col_topics)])
    
        for p in parties:
            info_party_temp = info_sliced[info_sliced.party == p]
            distr_temp_party = distr[distr.speakerID.isin(info_party_temp.memberID)]

            ind = party_distr[party_distr.party == p].index
        
            party_distr.loc[ind,col_topics] = func_party_probs(distr_temp_party, 
                                                               col_topics)
 
        party_distr.to_csv(myPath + "hansard_results_lda\\" + files_distr[y][:9] +
                        "_top_distr_party.csv")   
