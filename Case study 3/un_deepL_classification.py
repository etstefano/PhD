# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 14:31:43 2016

@author: stef
"""

import sys  
reload(sys)  
sys.setdefaultencoding('utf8')
from __future__ import print_function


import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split

import tensorflow as tf
import tensorflow.contrib.learn as skflow
from tensorflow.contrib.skflow import monitors
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score

import iso3166 
plt.style.use('ggplot')

np.random.seed(89)


myPath = "C:\\Users\\S\\Documents\\Uni\\UNspeeches\\"



ip_kpca = pd.read_csv(myPath + "ip_kpca.csv", index_col = 0)

#set train and test sets for all 150 feature
X_train, X_test, y_train, y_test = train_test_split(
    ip_kpca[ip_kpca.columns[35:-6]].values, 
    ip_kpca.cluster.values, test_size=0.2, random_state=0)

float(len(vec[vec==0]))/float(len(vec))
###############################################################################
#deep learning
# Build 3 layer DNN with 10, 20, 10 units respectively.
classifier = skflow.TensorFlowDNNClassifier(hidden_units=[100, 200, 100, 200],
    n_classes=4, steps=500)

# Fit and predict.
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
score = accuracy_score(y_test, classifier.predict(X_test))
print('Accuracy: {0:f}'.format(score))