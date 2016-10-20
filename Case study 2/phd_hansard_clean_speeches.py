# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 19:28:29 2015

@author: S
"""
import sys  
reload(sys)  
sys.setdefaultencoding('utf8')

import os
import pandas as pd
import ast
import nltk
#set higher column width, otherwise not all text is stored
pd.options.display.max_colwidth = 100000 

#set path accordingly
myPath = "C:\\Users\\S\\Documents\\Uni\\PhD\\final_thesis\\data\\hansard_input_lda\\"

#get list of files
speech_files = list()
for file in os.listdir(myPath):
    if "hansard_filt_cleaned" in file:
        speech_files.append(file)

###############################################################################
#set paramenters

topwords_pct = 0.1 #percentage of most frequent words to be dropped
least_common_w = 3 #drop words appearing less than this var
speech_thres = 5 #drop processed speeches featuring less than this var

###############################################################################
def clean_speeches(h_tagged):
    #this function gets rid of stopwords, words that appear less than 5 times, 
    #and top 0.05 most frequent words

    #pandas doesn't recognise the pos tagged columns as a list of tuples, let's
    #fix that
    func_tuple = lambda x: list(ast.literal_eval(x))
    h_tagged.speech_tagged = h_tagged.speech_tagged.apply(func_tuple)  
    
    #selects only nouns and adjectives
    print "selecting nouns and adjectives..."
    func_pos = lambda x: [w for w,tag in x if tag.startswith('N') or
                        tag.startswith('J')]
    h_tagged.speech_tagged = h_tagged.speech_tagged.apply(func_pos)    
    
    #get as a list of list 
    list_speeches = [s for s in h_tagged.speech_tagged.tolist()]
    
    #remove top 0.1%words and words appearing less than 3 times
    print "removing topwords and least common words..."
    frq = nltk.FreqDist([item for sublist in list_speeches for item in sublist])
    most_commons = [w for w,f in frq.most_common(n = int((topwords_pct/100)*len(frq)))]
    least_commons = [w for w,f in frq.viewitems() if f > least_common_w]

    func_rmv = lambda x: [w for w in x if w not in most_commons or w not in least_commons]  
    h_tagged.speech_tagged = h_tagged.speech_tagged.apply(func_rmv) 
    
    #stem words 
    print "stemming..."
    stemmer = nltk.SnowballStemmer("english")    
    func_stem = lambda x: [stemmer.stem(w) for w in x]
    h_tagged.speech_tagged = h_tagged.speech_tagged.apply(func_stem)     
    
   
    return h_tagged.speech_tagged

###############################################################################

###############################################################################
def create_docs(h_tagged, filename):
    #this function create the text files needed to write th dtm input
    
    punct = "—()[]"
    docs_per_session = list()
    list_index = list()

    #get session and filter by session
    sessions = sorted(h_tagged.sessionID.unique().tolist())
    for s in sessions: 
        
        print "exporting docs for session " + str(s)
        h_temp = h_tagged[h_tagged.sessionID == s] 
        
        docs_per_session.append(len(h_temp)) 
        
        with open(myPath + filename[:9] + '_all_speeches.txt', 'a') as f:
        #with open(myPath + 'speech_by_session\\' + str(int(s)) +'_speeches.txt', 'a') as f:
            for i in h_temp.speech_tagged.index:
                list_index.append(i)
                for word in h_temp.speech_tagged[i]:
                    if word not in punct:
                        f.write(word + " ")
                f.write('\n')
                
    with open(myPath + filename[:9] + '_seq.txt', 'w') as f: 
       for n in docs_per_session:
            f.write(str(n) + '\n')
            
    with open(myPath + filename[:9] + '_list_index.txt', 'w') as f: 
       for n in list_index:
            f.write(str(n) + ',' + str(h_tagged.sessionID[n]) + ', ' + 
            str(h_tagged.speakerID[n]) + ',' + str(h_tagged.date[n]) + '\n')
###############################################################################

#import dataset, change path accordingly

for filename in speech_files:
    h_tagged = pd.read_csv(myPath + filename)
    h_tagged.speech_tagged = clean_speeches(h_tagged)

    #remove empty speeches
    h_tagged =  h_tagged[h_tagged.speech_tagged.str.len() > speech_thres]

    #create text files needed for dtm input
    create_docs(h_tagged, filename)