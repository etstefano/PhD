# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 12:50:15 2015

This script creates the transfer entropy matrix for each speaker for each
range of dates considered

@author: S
"""

import sys  
reload(sys)  
sys.setdefaultencoding('utf8')

import os
import pandas as pd
import numpy as np
from jpype import *

import random
random.seed(8)

###############################################################################
myPath = "C:\\Users\\S\\Documents\\Uni\\PhD\\final_thesis\\data\\"
n_min_timestamps = 5
time_lag = 8 #days of lag + 1
n_shuffle_test = 200
#get list of files
files_distr = list()
for file in os.listdir(myPath + "hansard_te\\"):
    if "timeseries" in file:
        files_distr.append(file)
###############################################################################
#start the JVM
jarLocation = "C:\\Users\\S\\Documents\\Uni\\PhD\\final_thesis\\code\\infodynamics\\infodynamics.jar"
# Start the JVM (add the "-Xmx" option with say 1024M if you get crashes due to not enough memory space)
startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation)
teCalcClass = JPackage("infodynamics.measures.continuous.kernel").TransferEntropyCalculatorKernel
teCalc = teCalcClass()
teCalc.setProperty("NORMALISE", "false") # Normalise the individual variables
###############################################################################
def f_calc_te(destination, origin, n_min_timestamps, time_lag, dates):
    #for testing: dest index 10152, origin 10157 
#    dest = time_mat.ix[time_mat.index[dest]]
#    source = time_mat.ix[time_mat.index[source]]
    
    if len(destination[destination.notnull()]) < n_min_timestamps or len(origin[origin.notnull()]) < n_min_timestamps:
#        with open(myPath + "hansard_te\\log.txt", "a") as l:
#            l.write("timeseries length too short for " + str(destination.name) + 
#                    " or for " + str(origin.name) + "\n")
        return 0
    else:
        destination = destination[destination.notnull()]
        origin = origin[origin.notnull()]
        
        dest_vec = list()
        source_vec = list()
        #find closest date for each date of source
        
        for d in range(0,len(dates)):
            appended = False
            if dates[d] in origin.index:
                for lag in range(1,time_lag):
                    try:
                        if appended == False:
                            dest_vec.append(destination.ix[dates[d+lag]])
                            source_vec.append(origin.ix[dates[d]])                            
                    except (KeyError,IndexError):
                        continue
                    appended = True
                            
        
        if len(dest_vec) > 2:
            shuffle_vec = list()
            teCalc.initialise(1, 0.5) # Use history length 1 (Schreiber k=1), kernel width of 0.5 normalised units
            teCalc.setObservations(np.array(source_vec), np.array(dest_vec))
            # For copied source, should give something close to 1 bit:
            te = teCalc.computeAverageLocalOfObservations()
            for ns in range(0,n_shuffle_test): 
                source_vec_shuffled = np.array(source_vec)
                dest_vec_shuffled = np.array(dest_vec)
                np.random.shuffle(source_vec_shuffled)
                np.random.shuffle(dest_vec_shuffled)
                teCalc.setObservations(source_vec_shuffled,dest_vec_shuffled)
                shuffle_vec.append(teCalc.computeAverageLocalOfObservations())
                
            return (origin.name, destination.name, te, np.mean(np.array(shuffle_vec)),
                    np.median(np.array(shuffle_vec)), np.percentile(np.array(shuffle_vec),5),
                    np.percentile(np.array(shuffle_vec),95), np.percentile(np.array(shuffle_vec),10),
                    np.percentile(np.array(shuffle_vec),90),np.percentile(np.array(shuffle_vec),20),
                    np.percentile(np.array(shuffle_vec),80), len(dest_vec))
                    
        else:
            return 0
###############################################################################
###############################################################################

        
for y in range(0,len(files_distr))[1:]:
    print y
    
    with open(myPath + "hansard_te\\" + files_distr[y][:9] + "_te_edgelist_insight.csv", "a") as edge_list:
        edge_list.write("destination,source,te,te_mean,te_median,te_5pct,te_95pct,te_10pct,te_90pct,te_20pct,te_80pct,t_steps" +
                        "\n")
    
    print "working on years " + files_distr[y][:9]
        
    time_mat = pd.read_csv(myPath + "hansard_te\\" + files_distr[y], index_col = 0)
    
    dates = time_mat.columns
    te_adj = pd.DataFrame(index = time_mat.index, columns = time_mat.index)

    #fill transfer entropy adjacency matrix
    for dest in range(0,len(time_mat.index)):
        for source in range(0,len(time_mat.index)):
            print dest, source
            if dest != source:
                destination = time_mat.ix[time_mat.index[dest]]
                origin = time_mat.ix[time_mat.index[source]]
                
                te = f_calc_te(destination, origin,n_min_timestamps, time_lag,dates)
                if te == 0:
                    te_adj.loc[time_mat.index[dest],time_mat.index[source]] = te
                else:
                    te_adj.loc[time_mat.index[dest],time_mat.index[source]] = te[2]
                    with open(myPath + "hansard_te\\" + files_distr[y][:9] + "_te_edgelist_insight.csv", "a") as edge_list:
                        for value in te:
                            edge_list.write(str(value)+",")
                        edge_list.write("\n")
                    
    
    te_adj.to_csv(myPath + "hansard_te\\" + files_distr[y][:9] + "_te_adj.csv")
    
     rabber93@libero.it