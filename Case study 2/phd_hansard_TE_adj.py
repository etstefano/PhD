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
def f_calc_te(destination, origin, n_min_timestamps, time_lag):
    #for testing: dest index 10152, source 10157 
#    dest = time_mat.ix[time_mat.index[dest]]
#    source = time_mat.ix[time_mat.index[source]]
    
    if len(destination[destination.notnull()]) < n_min_timestamps or len(origin[origin.notnull()]) < n_min_timestamps:
        with open(myPath + "hansard_te\\log.txt", "a") as l:
            l.write("timeseries length too short for " + str(dest.name) + 
                    " or for " + str(source.name) + "\n")
        return 0
    else:
        destination = destination[destination.notnull()]
        origin = origin[origin.notnull()]
        
        dest_vec = list()
        source_vec = list()
        #find closest date for each date of source
        appended = False
        for d in origin.index:
            print d
            ind = np.datetime64(pd.to_datetime(d))
            #ind = np.datetime64(d.replace("/","-"), "%D,%M,%Y")
            for days in range(1,time_lag):
                try:
                    if appended == False:
                        day = pd.to_datetime(str(ind)).strftime("%d/%m/%Y")
                        day_plus = pd.to_datetime(str(ind+np.timedelta64(days, 'D'))).strftime("%d/%m/%Y")
                        dest_vec.append(destination.ix[day_plus])
                        source_vec.append(origin.ix[day])
                        appended = True
                except KeyError:
                        continue
            appended = False 
        
        if len(dest_vec) > 2:
            teCalc.initialise(1, 0.5) # Use history length 1 (Schreiber k=1), kernel width of 0.5 normalised units
            teCalc.setObservations(np.array(source_vec), np.array(dest_vec))
            # For copied source, should give something close to 1 bit:
            te = teCalc.computeAverageLocalOfObservations()

            return te
            
        else:
            return 0
###############################################################################
###############################################################################

        
for y in range(0,len(files_distr))[:1]:
    
    print "working on years " + files_distr[y][:9]
        
    time_mat = pd.read_csv(myPath + "hansard_te\\" + files_distr[y], index_col = 0)
    
    dates = time_mat.columns
    te_adj = pd.DataFrame(index = time_mat.index, columns = time_mat.index)

    #fill transfer entropy adjacency matrix
    for dest in range(0,len(time_mat.index))[:1]:
        for source in range(0,len(time_mat.index))[:2]:
            print dest, source
            if dest != source:
                destination = time_mat.ix[time_mat.index[dest]]
                origin = time_mat.ix[time_mat.index[source]]
                
                te_adj.loc[time_mat.index[dest],
                           time_mat.index[source]] = f_calc_te(destination, origin,
                                                    n_min_timestamps, time_lag)
                