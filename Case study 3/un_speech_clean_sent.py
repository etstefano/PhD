# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 11:18:01 2015
This script cleans UN speeches, by removing all punctuation but sentence definers, 
namely !?.:

@author: S
"""
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

import os
import nltk
from nltk.corpus import stopwords
import string

#set path
myPath = "C:\\Users\\S\\Documents\\Uni\\UNspeeches\\"
files = list()

nums = "0123456789"
stemmer = nltk.SnowballStemmer("english")

#get file names
for file in os.listdir(myPath + "input\\"):
    if file.endswith(".txt"):
        files.append(file)

#get punctuation to be removed
chars = "!?.:"
punct = string.punctuation
for p in chars:
    punct = punct.replace(p,'')

for fn in files: 
    print "working on " + fn
    with open(myPath + "input\\" + fn) as text:
        text = text.read().encode('utf-8')
        
        text = text.lower()        
        
        #remove numbers
        for n in nums:
            text = text.replace(n, "")
            
        #remove punctuation
        for p in punct+'—':
            text = text.replace(p, "")
        
        #to prevent unicode errors
        text = unicode(text, errors = 'replace')        
        
        #tokenize
        text = nltk.word_tokenize(text)
               
        
        #stem words
        text = [stemmer.stem(w) for w in text]
        
        with open(myPath + "input_clean_sent\\" + fn, "a") as output:
            for word in text:
                output.write(word + " ")
            output.write("\n")




