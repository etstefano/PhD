# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 10:35:15 2015

@author: S
"""

import sys  
reload(sys)  
sys.setdefaultencoding('utf8')

import os
from jpype import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition

import random
random.seed(8)

###############################################################################
#set parameters
myPath = "C:\\Users\\S\\Documents\\Uni\\PhD\\final_thesis\\data\\"
nTopics = 20
###############################################################################
#get list of files
files_distr = list()
for file in os.listdir(myPath + "hansard_results_lda\\"):
    if "topic_distr" in file:
        files_distr.append(file)
        
###############################################################################

for y in range(0,len(files_distr))[:1]:
    
    print "working on years " + files_distr[y][:9]
    
    #import dataframe
    distr = pd.read_csv(myPath + "hansard_results_lda\\" + files_distr[y], 
                        index_col = 0)
    
    dates = distr.date.unique()
    spd = [len(distr[distr.date == i]) for i in dates]
    col_topics = distr.columns[4:]

    dist_mat = distr[col_topics].values    
    
    model = decomposition.RandomizedPCA(n_components = 1)
    a = model.fit_transform(dist_mat)

    
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(a[:,0], a[:,1], a[:,2])
    
    
    source = np.random.choice(a.reshape(len(a)), size=300, replace=False, p=None)    
    dest = np.concatenate((np.array([0]), np.random.choice(a.reshape(len(a)), 
                                    size=299, replace=False, p=None)))
    
from jpype import *
import random
import math
import numpy as np

# Change location of jar to match yours:
jarLocation = "C:\\Users\\S\\Documents\\Uni\\PhD\\final_thesis\\code\\infodynamics\\infodynamics.jar"
# Start the JVM (add the "-Xmx" option with say 1024M if you get crashes due to not enough memory space)
startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation)

# Generate some random normalised data.
numObservations = 1000
covariance=0.4
# Source array of random normals:
sourceArray = [random.normalvariate(0,1) for r in range(numObservations)]
# Destination array of random normals with partial correlation to previous value of sourceArray
destArray = [0] + [sum(pair) for pair in zip([covariance*y for y in sourceArray[0:numObservations-1]], \
                                             [(1-covariance)*y for y in [random.normalvariate(0,1) for r in range(numObservations-1)]] ) ]
# Uncorrelated source array:
sourceArray2 = [random.normalvariate(0,1) for r in range(numObservations)]
# Create a TE calculator and run it:
teCalcClass = JPackage("infodynamics.measures.continuous.kernel").TransferEntropyCalculatorKernel
teCalc = teCalcClass()
teCalc.setProperty("NORMALISE", "true") # Normalise the individual variables
teCalc.initialise(1, 0.5) # Use history length 1 (Schreiber k=1), kernel width of 0.5 normalised units
teCalc.setObservations(np.array(source), np.array(dest))
# For copied source, should give something close to 1 bit:
result = teCalc.computeAverageLocalOfObservations()
print("TE result %.4f bits; expected to be close to %.4f bits for these correlated Gaussians but biased upwards" % \
    (result, math.log(1/(1-math.pow(covariance,2)))/math.log(2)))
teCalc.initialise() # Initialise leaving the parameters the same
teCalc.setObservations(np.array(sourceArray2), np.array(destArray))
# For random source, it should give something close to 0 bits
result2 = teCalc.computeAverageLocalOfObservations()
print("TE result %.4f bits; expected to be close to 0 bits for uncorrelated Gaussians but will be biased upwards" % \
    result2)
    
###############################################################################
