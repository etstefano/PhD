# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 14:42:21 2015

This script takes unstructured data from the House of Commons speeches dataset, 
and cleans it. Next step is to use the clean data to run topic modeling. 

@author: Stefano Gurciullo
"""

# -*- coding: utf-8 -*-
import sys  
reload(sys)  
sys.setdefaultencoding('utf8')


#libraries needed for text cleaning
import pandas as pd
import nltk
from nltk.corpus import stopwords
import string

pd.options.display.max_colwidth = 100000 #set column width, otherwise not all text is seen

###############################################################################
myPath_input = "C:\\Users\\S\\Documents\\Uni\\Hansard\\"
myPath_output = "C:\\Users\\S\\Documents\\Uni\\PhD\\final_thesis\\data\\hansard_input_lda\\"

###############################################################################
def clean_speech(speech):
    
    print "lowering cases... \n"
    speech = speech.str.lower()
    
#    print "removing punctuation... \n"
#    for punct in string.punctuation+'—':
#        speech = speech.str.replace(punct, "")
       
    
    for n in range(0,len(speech)):
        try:
            print "tokenizing speech " + str(n)
            speech.iloc[n] = nltk.word_tokenize(speech.iloc[n])
        except TypeError:
            continue
        
#        speech.iloc[n] = [w for w in speech.iloc[n] if w not in string.punctuation+'—']        
#        
#        print "removing stopwords in speech " + str(n)
#        speech.iloc[n] = [w for w in speech.iloc[n] if w not in stopwords.words('english')]
        
#        print "removing political stopwords in speech " + str(n)
#        speech.iloc[n] = [w for w in speech.iloc[n] if w not in polstop]
        
        speech.iloc[n] = [unicode(w, errors='ignore') for w in speech.iloc[n]]
        
        
        speech.iloc[n] = nltk.pos_tag(speech.iloc[n]) #remove this if the code 
        #below is uncommented
        
#        try:
#            speech_pos = nltk.pos_tag(speech.iloc[n])
#            print "extracting only nouns in speech " + str(n)
#            speech.iloc[n] = [w for w,tag in speech_pos if tag.startswith('N') or
#                        tag.startswith('J')]
#        except TypeError:
#            continue
    return speech
################################################################################

################################################################################
def create_documents(h_temp, r):
    
    nums = "0123456789"
    stopchars = "[]\\"    

    print 'exporting docs for time series number ' + str(r)
        
    members = sorted([int(i) for i in h_temp.speakerID.unique().tolist()])
    h_temp = h_temp.speech.groupby(h_temp.speakerID)
                
    with open(myPath_output + 'all_speeches_' + str(ranges[r][0][0]) + '-' +
        str(ranges[r][1][0])+ '.txt', 'a') as f:
            #with open(myPath + 'speech_by_session\\' + str(int(s)) +'_speeches.txt', 'a') as f:
            for m in members:
                h = h_temp.get_group(m).to_string().encode('utf-8')
                h = h.replace('\n',',')
                h = h.replace("u'", '')
                for c in nums:
                    h = h.replace(c,'')
                for c in stopchars:
                    h = h.replace(c,'')
                h = h.replace(' ','')
                h = h.replace(',',' ')
                h = h.replace("'",'')
                f.write(h + '\n')
                
                with open(myPath_output + 'sequence_' + str(ranges[r][0][0]) + 
                    str(ranges[r][1][0])+ '.txt', 'a') as g: 
                        g.write(str(m) + '\n')

###############################################################################
##import entire dataset
hansard = pd.read_csv(myPath_input +
    "Hansard_speeches_1935-2014_SF.csv.bz2", header=None, na_values='\N',
    names=['sessionID', 'speakerID', 'number', 'date', 'speech'], 
    encoding = 'utf-8', compression='bz2')
hansard.speech = hansard.speech.str.encode('utf-8')
hansard.date = pd.to_datetime(hansard.date)

###############################################################################
#filter only data on the time of interest
#getting speeches made from day before election to one year before
ranges = [([1994, 4, 30], [1997, 4, 30]), #Tony Blair won - Labour
          ([1998, 6, 6], [2001, 6, 6,]), #Tony Blair won - Labour
          ([2002, 5, 4], [2005, 5, 4]), # Tony Blair/Gordon Brown - Labour
          ([2007, 5, 5], [2010, 5, 5]), #David Cameron - Conservative/LibDem
          ([2011, 2, 13], [2014, 2, 13])] #David Cameron - Conservative

for r in range(0,len(ranges)):
    rng = pd.date_range(start = pd.datetime(ranges[r][0][0], ranges[r][0][1],
                        ranges[r][0][2]), end = pd.datetime(ranges[r][1][0],
                        ranges[r][1][1], ranges[r][1][2]))
    
    #select speeches for the range of dates
    h_temp = hansard[hansard.date.isin(rng)]
    h_temp['speech_tagged'] = clean_speech(h_temp.speech)
    h_temp.to_csv(myPath_output + str(ranges[r][0][0]) + '-' +
                    str(ranges[r][1][0])+ "hansard_filt_cleaned.csv")


    create_documents(h_temp, r)

bu = h_temp.speech[:20]
a = clean_speech[hansard.speech]
b = nltk.help.upenn_tagset()
