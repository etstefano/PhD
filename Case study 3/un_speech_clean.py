# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 10:52:11 2015

@author: S
"""
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

import os
import nltk
from nltk.corpus import stopwords
import string

###############################################################################
#set path
myPath = "C:\\Users\\S\\Documents\\Uni\\UNspeeches\\"
files = list()
###############################################################################
#set numbers and stemmer
nums = "0123456789"
stemmer = nltk.SnowballStemmer("english")

###############################################################################
#get file names
for file in os.listdir(myPath + "input\\"):
    if file.endswith(".txt"):
        files.append(file)

###############################################################################        
count = 0        
#clean each speech
for fn in files:
    with open(myPath + "input\\" + fn) as text:
        text = text.read().encode('utf-8')
        
        text = text.lower()        
        
        #remove numbers
        for n in nums:
            text = text.replace(n, "")
            
        #remove punctuation
        for punct in string.punctuation+'—':
            text = text.replace(punct, "")
        
        #tokenize
        text = nltk.word_tokenize(text)
        
        #remove stopwords
        text = [w for w in text if w not in stopwords.words('english')]
        
        text = [unicode(w, errors='ignore') for w in text]        
        
        #stem words
        text = [stemmer.stem(w) for w in text]
        
        with open(myPath + "input_clean\\" + fn, "a") as output:
            for word in text:
                output.write(word + " ")
            output.write("\n")
        
        count = (count + 1)*100
        print str(round(count/len(files),2)) + "% done"
        