# -*- coding: utf-8 -*-
"""
Created on Mon Apr 04 15:52:45 2016

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


sentences = list()

for fn in files: 
    with open(myPath + "input_clean2\\" + fn) as text:
        sentences.append(doc2vec.LabeledSentence(words=text.read().split(' ')[:-1], labels=[fn[:-4]]))

model = doc2vec.Doc2Vec(sentences, size=150, window=10, min_count=5, workers=4)
model.save(myPath + "doc2vec\\" + 'un_speeches.doc2vec')

###############################################################################
#Load model

model = doc2vec.Doc2Vec.load(myPath + "doc2vec\\" + 'un_speeches.doc2vec')  
#build dataframe with the results
cols = np.shape(model.syn0)[1]
data = pd.DataFrame(index = model.vocab.keys(), columns = range(0,cols), 
                    data = model.syn0, dtype = float)
                    
data.to_csv(myPath + "doc2vec\\" + "doc2vec_results.csv")

#build adjancency list for all terms and doc, then export the matrix
vocab = pd.Series(model.vocab.keys())
vocab.to_csv(myPath + "doc2vec\\" + "vocab_index.csv")

meta_mat = np.zeros([len(vocab), len(vocab)])
for i in range(1,len(vocab)):
    print i
    for j in range(i+1, len(vocab)):
        meta_mat[i,j] = model.n_similarity([vocab[i]], [vocab[j]])

np.savetxt(myPath + "doc2vec\\" + "meta_mat.csv", meta_mat)

meta_mat_pd = pd.DataFrame(data = meta_mat, index = vocab.keys, columns = vocab.keys())