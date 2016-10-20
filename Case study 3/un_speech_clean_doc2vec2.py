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
for file in os.listdir(myPath + "input2\\"):
    if file.endswith(".txt"):
        files.append(file)

###############################################################################        
count = 0        
#clean each speech
for fn in files:
    count += 1
    if count % 10 == 0:
        print str(count) + " docs out of " + str(len(files)) + " processed"
        
    with open(myPath + "input2\\" + fn) as text:
        text = text.read().encode('utf-8')
        
        text = text.lower()        
        
        #remove numbers
        for n in nums:
            text = text.replace(n, "")
            
        #remove punctuation
        #for punct in string.punctuation+'—':
         #   text = text.replace(punct, "")
        
        text = text.replace("..", ".")
        
        text = unicode(text, errors='replace')        
        
        #tokenize
        text = nltk.word_tokenize(text)
        
        #remove stopwords
        #text = [w for w in text if w not in stopwords.words('english')]
        
        #text = [unicode(w, errors='ignore') for w in text]        
        
        #stem words
        text = [stemmer.stem(w) for w in text]
        
        #if it starts with a number, delete it
#        if text[0][0]  in str(nums):
#            text.pop(0)
        
        with open(myPath + "input_clean_doc2vec2\\" + fn, "a") as output:
            for word in text:
                output.write(word + " ")
            output.write("\n")
        

        