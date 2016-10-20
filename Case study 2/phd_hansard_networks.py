# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 17:35:53 2015

@author: S
"""


import sys  
reload(sys)  
sys.setdefaultencoding('utf8')

import os
import pandas as pd
import igraph as ig
import networkx as nx

import random
random.seed(8)

###############################################################################
myPath = "C:\\Users\\S\\Documents\\Uni\\PhD\\final_thesis\\data\\"

#get list of files
files_eg = list()
for file in os.listdir(myPath + "hansard_te\\"):
    if "edgelist" in file:
        files_eg.append(file)

files_adj = list()
for file in os.listdir(myPath + "hansard_te\\"):
    if "adj" in file:
        files_adj.append(file)

for y in range(0, len(files_eg))[:1]:
    
    adj = pd.read_csv(myPath + "hansard_te\\" + files_adj[y], index_col = 0)    
    
    el = pd.read_csv(myPath + "hansard_te\\" + files_eg[y])
    ell = el[(el.te <= el.te_5pct) | (el.te >= el.te_95pct)]
    ell = ell[el.te > 0.8]

    g = nx.DiGraph()
    g.add_nodes_from([int(n) for n in adj.index.values])  
    g.add_edges_from([(int(ell.source.ix[i]), int(ell.destination.ix[i])) for i in ell.index])
    
    pos = nx.spring_layout(g, k = 1)
    nx.draw_networkx_nodes(g, pos, node_size = 4, node_color = '#000066')
    nx.draw_networkx_edges(g, pos, width = 0.5, edge_color='#99CCFF', alpha = 0.5)
    
    nx.write_gml(g, myPath + "hansard_te\\" + "net.gml")
    
    nx.draw(g)
    
   