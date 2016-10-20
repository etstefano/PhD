# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 11:04:19 2015

@author: S
"""
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from gensim import corpora, models, similarities
import os

myPath = "C:\\Users\\S\\Documents\\Uni\\UNspeeches\\"
files = list()
for file in os.listdir(myPath + "input_clean\\"):
    if file.endswith(".txt"):
        files.append(file)
        
years = [str(i) for i in range(1971, 2015)]
speech_by_y = []
n_speech_y = []

with open(myPath + "input_clean\\un_all_speeches.txt", "a") as f:
    for y in years:
        sublist = [speech for speech in files if y in speech]
        speech_by_y = speech_by_y + sublist
        n_speech_y.append(len(sublist))
        for speech in sublist:
            with open(myPath + "input_clean\\" + speech) as s:
                s = s.read()
                f.write(s)
                #f.write('\n')

with open (myPath + "input_clean\\list_speeches.txt", "a") as f:
    for nation in speech_by_y:
        f.write(nation + "\n")

with open (myPath + "input_clean\\speech_per_year.txt", "a") as f:
    for num in n_speech_y:
        f.write(str(num) + "\n")

class MyCorpus(object):
     def __iter__(self):
         for f in open(myPath + "input_clean\\un_all_speeches.txt"):
             # assume there's one document per line, tokens separated by whitespace
             yield dictionary.doc2bow(f.split(' '))
    
corpus = MyCorpus()

for vector in corpus[:20]: # load one vector into memory at a time
     print(vector)


