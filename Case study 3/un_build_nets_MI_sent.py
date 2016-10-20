# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 19:07:51 2015

@author: S
"""

import pandas as pd
import networkx as nx
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.metrics.cluster import normalized_mutual_info_score

myPath = "C:\\Users\\S\\Documents\\Uni\\UNspeeches\\"

###############################################################################
def get_mutual_information(docs, i,j):
    
    t1 = docs[docs.columns[i]].values
    t2 = docs[docs.columns[j]].values
    
    mi_score = normalized_mutual_info_score(t1,t2)
    
    return mi_score
###############################################################################

files_sent = list()
for file in os.listdir(myPath + "output_sent\\"):
    if "sentiment" in file:
        files_sent.append(file)

files_topic = list()
for file in os.listdir(myPath + "output_lda\\"):
    if "docs" in file:
        files_topic.append(file)

for f in range(0,len(files_topic)): 
    
    docs_sent = pd.read_csv(myPath + "output_sent\\" + files_sent[f], index_col = 0,
                       encoding = "utf-8")
    docs_top  = pd.read_csv(myPath + "output_lda\\" + files_topic[f], index_col = 0,
                       encoding = "utf-8")
    
    #calculate adj matrix with distances
    adj = np.zeros((len(docs_sent.columns), len(docs_sent.columns)))
    
    for i in range(0, len(docs_sent.columns)):
        for j in range(i+1, len(docs_sent.columns)):       
            adj[i,j] = np.mean([get_mutual_information(docs_sent,i,j), 
                                    get_mutual_information(docs_top,i,j)])
            
    np.savetxt(myPath + "networks_mi_sent\\" + files_topic[f][:4]+ "_adj.csv", 
               adj, fmt='%.18f', delimiter=",")
               
    
###############################################################################
#create networks
    
files = list()
for file in os.listdir(myPath + "networks_mi_sent\\"):
    if file.endswith(".csv"):
        files.append(file)   
        
countries = list()
for file in os.listdir(myPath + "input_clean\\"):
    countries.append(file)  
        
for fname in files: 
    print fname
    adj = np.genfromtxt(myPath + "networks_mi_sent\\" + fname, delimiter=',') 
    #threshold = 0.8  
    threshold = np.percentile(adj, 90) #50% of matrix is zero, here we are selecting
                                        # (65-50)*2 = 30% threshold
 
    c_names = [c[:3] for c in countries if fname[:4] in c]
    labels = dict()
    for c in range(0,len(c_names)):
        labels[c] = c_names[c]

    net = nx.from_numpy_matrix(adj)
    n_bunch = [(u,v,d) for (u,v,d) in net.edges(data=True) if d['weight'] < threshold]
    net.remove_edges_from(n_bunch)
    nx.set_node_attributes(net,"label",labels)    
    
    nx.write_gml(net, myPath + "networks_mi_sent\\" + fname[:4] + "_net_top20.gml")
    
    pos = nx.spring_layout(net, k = 0.1)
    nx.draw_networkx_nodes(net, pos, node_size = 0, node_color = '#000066')
    nx.draw_networkx_edges(net, pos, width = 0.5, edge_color='#99CCFF', alpha = 0.5)
    nx.draw_networkx_labels(net,pos,labels,font_size=5)
    plt.axis('off')
    
    plt.savefig(myPath + "networks_mi_sent\\" + fname[:4] + "_net",dpi=500)
    plt.close()
