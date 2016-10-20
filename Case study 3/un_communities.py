# -*- coding: utf-8 -*-
"""
Created on Tue May 26 16:15:55 2015

@author: S
"""
import igraph as ig
import os
import pandas as pd

myPath = "C:\\Users\\S\\Documents\\Uni\\UNspeeches\\"

files = list()
for file in os.listdir(myPath + "networks\\"):
    if file.endswith(".gml"):
        files.append(file)

comms = dict()
for net in files:
    print net[:4]
    #get list of countries for this year
    countries = list()
    for file in os.listdir(myPath + "input_clean\\"):
        if net[:4] in file:
            countries.append(file[:3])
    #import graph        
    g = ig.read(myPath + "networks\\" + net)
    #label each node with country codes
    g.vs['country'] = countries
    #get new weights such that the higher the value, the stronger the connection
    avg = sum(g.es['weight']) / float(len(g.es['weight']))
    #g.es['new_weights'] = [1/w for w in g.es['weight']]
    g.es['new_weights'] = [avg/w for w in g.es['weight']]


    #get communities
    comms[int(net[:4])] = dict()
    C = g.community_infomap(edge_weights= g.es['new_weights'], trials=15)
    for c in range(0,len(C)):
        comms[int(net[:4])][c] = dict()
        comms[int(net[:4])][c]['id'] = C[c]
        comms[int(net[:4])][c]['iso'] = [countries[n] for n in C[c]]
        

with open(myPath + "networks\\" + "communities_ISOcodes1.txt", "a") as f:
    for key in comms.keys():
        f.write(str(key) + "\n")
        for subkey in comms[key].keys():
            f.write(str(subkey) + ": ")
            for i in comms[key][subkey]['iso']:
                f.write(i + ',')
            f.write('\n')
        f.write('\n')
        f.write('\n')
                

communities = pd.DataFrame.from_dict(comms)
communities.to_csv(myPath + "networks\\communities1.csv")
