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

from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity

from gensim.models import doc2vec
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

myPath = "C:\\Users\\S\\Documents\\Uni\\UNspeeches\\"

files = list()
for file in os.listdir(myPath + "input_clean_doc2vec2\\"):
    if file.endswith(".txt"):
        files.append(file)


sentences = list()

for fn in files: 
    with open(myPath + "input_clean_doc2vec2\\" + fn) as text:
        sentences.append(doc2vec.LabeledSentence(words=text.read().split(' ')[:-1], labels=[fn[:-4]]))

model = doc2vec.Doc2Vec(sentences, size=200, window=10, min_count=5, workers=4)
model.save(myPath + "doc2vec2\\" + 'un_speeches2.doc2vec')

###############################################################################
#Load model

model = doc2vec.Doc2Vec.load(myPath + "doc2vec2\\" + 'un_speeches2.doc2vec')  
#build dataframe with the results
cols = np.shape(model.syn0)[1]
data = pd.DataFrame(index = model.vocab.keys(), columns = range(0,cols), 
                    data = model.syn0, dtype = float)
                    
data.to_csv(myPath + "doc2vec2\\" + "doc2vec2_results.csv")

#make meta_mat of states
import re
def hasNumbers(inputString):
    return bool(re.search(r'\d', inputString))
states = [s for s in model.vocab.keys() if hasNumbers(s)]
meta_mat = np.zeros([len(states), len(states)])


for ss in range(0,len(states)):
    if ss%10 == 0:
        print str(ss) + " docs done out of " + str(len(states))
    for cc in range(ss+1, len(states)):
        meta_mat[ss,cc] = model.n_similarity([states[ss]], [states[cc]])

meta_mat_df = pd.DataFrame(data = meta_mat, index = states, columns = states)
meta_mat_df.to_csv(myPath + "doc2vec2\\" + "meta_mat.csv")

###############################################################################
###############################################################################
#define and use function to extract country similarities with certain concepts
env_words = ['environ', 'chang', 'climat']
ter_words = ['terror', 'islam']
edu_words = ['educ', 'school']
hea_words = ['sanit', 'health']
nuc_words = ['nuclear', 'weapon']

env_series = f_extract_similarity(env_words, model, colnames = ["env_similarity"])
ter_series = f_extract_similarity(ter_words, model, colnames = ["ter_similarity"])

edu_series = f_extract_similarity(edu_words, model, colnames = ["edu_similarity"])
hea_series = f_extract_similarity(hea_words, model, colnames = ["hea_similarity"])

nuc_series = f_extract_similarity(nuc_words, model, colnames = ["nuc_similarity"])

def f_extract_similarity(words, model, colnames = [""]):
    states = [s for s in model.vocab.keys() if hasNumbers(s)]
    distances = np.zeros(len(states))
    for ss in range(0,len(states)):
            if ss%10 == 0:
                print(str(ss) + " docs done out of " + str(len(states)))
            distances[ss] = model.n_similarity([states[ss]], words)
    return pd.DataFrame(data = distances, index = states, columns = colnames)
    
data = pd.read_csv(myPath + "doc2vec2\\" + "doc2vec_kpca.csv", index_col = 0)
data = data.join(env_series)
data = data.join(ter_series)

data = data.join(edu_series)
data = data.join(hea_series)

data = data.join(nuc_series)

data.to_csv(myPath + "doc2vec2\\" + "doc2vec_kpca_2.csv")
