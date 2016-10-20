# -*- coding: utf-8 -*-
"""
Created on Tue May 12 10:34:22 2015

@author: S
"""

import pandas as pd
import networkx as nx
import numpy as np
import math
import matplotlib.pyplot as plt
import os

###############################################################################
def calc_dist(docs,i,j):
    
    t1 = docs[docs.columns[i]].values
    t2 = docs[docs.columns[j]].values
    
    dist = math.sqrt((t1[0]-t2[0])**2 + (t1[1]-t2[1])**2 + (t1[2]-t2[2])**2 +
                        (t1[3]-t2[3])**2 + (t1[4]-t2[4])**2 + (t1[5]-t2[5])**2
                        + (t1[6]-t2[6])**2 + (t1[7]-t2[7])**2)
                        
    return dist
###############################################################################

myPath = "C:\\Users\\S\\Documents\\Uni\\UNspeeches\\"

files = list()
for file in os.listdir(myPath + "output_lda\\"):
    if "docs" in file:
        files.append(file)

for fname in files: 
    
    docs = pd.read_csv(myPath + "output_lda\\" + fname, index_col = 0,
                       encoding = "utf-8")
    
    #calculate adj matrix with distances
    adj = np.zeros((len(docs.columns), len(docs.columns)))
    
    for i in range(0, len(docs.columns)):
        for j in range(i+1, len(docs.columns)):       
            adj[i,j] = calc_dist(docs,i,j)
            
    np.savetxt(myPath + "networks\\" + fname[:4]+ "_adj.csv", 
               adj, fmt='%.18f', delimiter=",")

    
###############################################################################
#create networks
    
files = list()
for file in os.listdir(myPath + "networks_mi\\"):
    if file.endswith(".csv"):
        files.append(file)   
        
countries = list()
for file in os.listdir(myPath + "input_clean\\"):
    countries.append(file)  
        
for fname in files: 
    print fname
    adj = np.genfromtxt(myPath + "networks_mi\\" + fname, delimiter=',') 
    threshold = 0.8  
    #threshold = np.percentile(adj, 52.5) #50% of matrix is zero, here we are selecting
                                        # (65-50)*2 = 30% threshold
 
    c_names = [c[:3] for c in countries if fname[:4] in c]
    labels = dict()
    for c in range(0,len(c_names)):
        labels[c] = c_names[c]

    net = nx.from_numpy_matrix(adj)
    n_bunch = [(u,v,d) for (u,v,d) in net.edges(data=True) if d['weight'] < threshold]
    net.remove_edges_from(n_bunch)
    
    nx.write_gml(net, myPath + "networks_mi\\" + fname[:4] + "_net.gml")
    
    pos = nx.spring_layout(net, k = 0.1)
    nx.draw_networkx_nodes(net, pos, node_size = 5, node_color = '#000066')
    nx.draw_networkx_edges(net, pos, width = 0.5, edge_color='#6633CC')
    nx.draw_networkx_labels(net,pos,labels,font_size=5)
    plt.axis('off')
    
    plt.savefig(myPath + "networks_mi\\" + fname[:4] + "_net",dpi=500)
    plt.close()


                        
    