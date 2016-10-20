# -*- coding: utf-8 -*-
"""
Created on Tue May 26 16:15:55 2015

@author: S
"""
import igraph as ig
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

myPath = "C:\\Users\\S\\Documents\\Uni\\UNspeeches\\"

files = list()
for file in os.listdir(myPath + "networks_mi\\"):
    #if file.endswith(".gml"):
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
    g = ig.read(myPath + "networks_mi\\" + net)
    #label each node with country codes
    g.vs['country'] = countries

    #get communities
    comms[int(net[:4])] = dict()
    C = g.community_infomap(edge_weights= g.es['weight'], trials=15)
    for c in range(0,len(C)):
        comms[int(net[:4])][c] = dict()
        comms[int(net[:4])][c]['id'] = C[c]
        comms[int(net[:4])][c]['iso'] = [countries[n] for n in C[c]]
        

with open(myPath + "networks_mi\\" + "communities_ISOcodes_top20.txt", "a") as f:
    for key in comms.keys():
        f.write(str(key) + "\n")
        for subkey in comms[key].keys():
            f.write(str(subkey) + ": ")
            for i in comms[key][subkey]['iso']:
                f.write(i + ',')
            f.write('\n')
        f.write('\n')
        f.write('\n')
                

year_col = []
frames = []

for year, communities in comms.iteritems():
    year_col.append(year)
    frames.append(pd.DataFrame.from_dict(communities, orient='index'))
    
a = pd.concat(frames, keys=year_col)
a.to_csv(myPath + "networks_mi\\communities_top20.csv")

###############################################################################
#Get structural properties of graphs

###############################################################################
def movingaverage (values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
return sma   
###############################################################################

cols = ['density', 'diameter', 'average path length', 'clustering coeff']
index = range(1970, 2015)

struct_prop = pd.DataFrame(index = index, columns = cols)


for net in files:
    year = int(net[:4])    
    print net[:4]
    #get list of countries for this year
    countries = list()
    for file in os.listdir(myPath + "input_clean\\"):
        if net[:4] in file:
            countries.append(file[:3])
    #import graph        
    g = ig.read(myPath + "networks_mi\\" + net)
    #label each node with country codes
    g.vs['country'] = countries
    
    struct_prop.loc[year, 'density'] = g.density()
    struct_prop.loc[year, 'diameter'] = g.diameter()
    struct_prop.loc[year, 'average path length'] = g.average_path_length()
    struct_prop.loc[year, 'clustering coeff'] = g.transitivity_undirected()
    
struct_prop.to_csv(myPath + 'networks_mi\\net_stats.csv')


ma_year = range(1975, 2011)
ma_cc = movingaverage (struct_prop['clustering coeff'], 10)
ma_d = movingaverage (struct_prop['density'], 10)
ma_apl = movingaverage (struct_prop['average path length'], 10)
    
a = np.asarray([ ma_year, ma_d, ma_cc, ma_apl ])
a = a.transpose()
np.savetxt(myPath + "networks_mi\\net_stats_ma.csv", a, delimiter=",")

plt.plot(ma_year, ma_d, linewidth = 2.0, color = '#000066')
plt.ylabel('Density - 10 yr moving average')
plt.xlabel('Year')
plt.show()
plt.savefig(myPath + "networks_mi\\" + "density_ma",dpi=500)
plt.close()

plt.plot(ma_year, ma_cc, linewidth = 2.0, color = '#336666')
plt.ylabel('Clustering coefficient - 10 yr moving average')
plt.xlabel('Year')
plt.show()
plt.savefig(myPath + "networks_mi\\" + "clustering_ma",dpi=500)
plt.close()

plt.plot(ma_year, ma_apl, linewidth = 2.0, color = '#333366')
plt.ylabel('Average path length - 10 yr moving average')
plt.xlabel('Year')
plt.show()
plt.savefig(myPath + "networks_mi\\" + "apl_ma",dpi=500)
plt.close()
