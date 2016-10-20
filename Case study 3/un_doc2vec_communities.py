# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 09:57:53 2016

@author: S
"""

import sys  
reload(sys)  
sys.setdefaultencoding('utf8')

import os 
import numpy as np
import pandas as pd
import igraph as ig
import iso3166 


myPath = "C:\\Users\\S\\Documents\\Uni\\UNspeeches\\"

#get list of all adj mats and indices 
adjs = list()
inds = list()
for file in os.listdir(myPath + "doc2vec\\"):
    if "adj" in file:
        adjs.append(file)
    elif "countries" in file:
        inds.append(file)

adjs = sorted(adjs)
inds = sorted(inds)

country_dict = iso3166.countries_by_alpha3
country_dict['CSK'] = iso3166.Country(name = u'Czechoslovakia', alpha2='CS', 
    alpha3='CSK', numeric='1000', apolitical_name=u'Czechoslovakia')
country_dict['YUG'] = iso3166.Country(name = u'Yugoslavia', alpha2='YU', 
    alpha3='YUG', numeric='1001', apolitical_name=u'Yugoslavia')
country_dict['YDY'] = iso3166.Country(name = u'South Yemen', alpha2='YDD', 
    alpha3='YDY', numeric='1002', apolitical_name=u'South Yemen')
country_dict['DDR'] = iso3166.Country(name = u'German Democratic Republic', alpha2='DD', 
    alpha3='DDR', numeric='1003', apolitical_name=u'German Democratic Republic')

#for each year, detect and visualise communities
for i in range(0,len(adjs)):
    #load adj
    adj = np.loadtxt(myPath + "doc2vec\\" + adjs[i])
    labels = pd.read_csv(myPath + "doc2vec\\" + inds[i], index_col = 0, 
                         names = ['labels'])
    #create graph
    thresh = np.mean(adj[adj.nonzero()]) #don't take into account links less than the mean
    g = ig.Graph.Adjacency((adj > thresh).tolist())
    g = g.as_undirected()
    g.es['weight'] = adj[adj > thresh]
    g.vs['label'] = [ll[:3] for ll in labels['labels'].values.tolist()]
    
    #identify communities using multivel algo
    #C = g.community_infomap(edge_weights= g.es['weight'], trials=20)
    #c1 = g.community_label_propagation(weights= g.es['weight'])
    c2 = g.community_multilevel(weights= g.es['weight'], return_levels = False)
    clusters = [ii[0] for ii in c2.as_cover().membership]
    
    #dataframe with membership 
    labels['membership'] = clusters
    labels['iso_code'] = labels['labels'].apply(lambda x: x[:3])
    labels['year'] = labels['labels'].apply(lambda x: x[-4:])
    names = [iso3166.countries_by_alpha3[code].name for code in labels.iso_code.values]
    labels['name'] = names
    
    labels.to_csv(myPath + "doc2vec\\communities_by_multilevel\\" + adjs[i][:4] + "_comms.csv")


###############################################################################
###############################################################################
#concatenate all commmunities csv in one dataframe
comms_files = list()
for file in os.listdir(myPath + "doc2vec\\communities_by_multilevel\\"):
    if "comms" in file:
        comms_files.append(file)

for cf in comms_files:
    if '1962' in cf:
        df = pd.read_csv(myPath + "doc2vec\\communities_by_multilevel\\" + cf, 
                         index_col = 0)
    else:
        df1 = pd.read_csv(myPath + "doc2vec\\communities_by_multilevel\\" + cf, 
                         index_col = 0)
                         
        df = pd.concat([df, df1])

df.to_csv(myPath + "doc2vec\\communities_by_multilevel\\" + "communities.csv")
    