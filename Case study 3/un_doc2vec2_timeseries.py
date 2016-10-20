# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 09:09:47 2016

This script checks similarity dynamics over time, against state-selves and the US.
It also does clustering

@author: S
"""

import sys  
reload(sys)  
sys.setdefaultencoding('utf8')

import os 
import numpy as np
import pandas as pd

import iso3166 

###############################################################################
#retrieve iso country labels
country_dict = iso3166.countries_by_alpha3
country_dict['CSK'] = iso3166.Country(name = u'Czechoslovakia', alpha2='CS', 
    alpha3='CSK', numeric='1000', apolitical_name=u'Czechoslovakia')
country_dict['YUG'] = iso3166.Country(name = u'Yugoslavia', alpha2='YU', 
    alpha3='YUG', numeric='1001', apolitical_name=u'Yugoslavia')
country_dict['YDY'] = iso3166.Country(name = u'South Yemen', alpha2='YDD', 
    alpha3='YDY', numeric='1002', apolitical_name=u'South Yemen')
country_dict['DDR'] = iso3166.Country(name = u'German Democratic Republic', alpha2='DD', 
    alpha3='DDR', numeric='1003', apolitical_name=u'German Democratic Republic')
###############################################################################

myPath = "C:\\Users\\S\\Documents\\Uni\\UNspeeches\\"

meta_mat = pd.read_csv(myPath + "doc2vec2\\"  + "meta_mat.csv", index_col = 0)

###############################################################################
#get iso codes, years and names
iso_codes = sorted(list(set([i[:3] for i in meta_mat.index.tolist()])))
years = sorted(list(set([i[-4:] for i in meta_mat.index.tolist()])))
names = [iso3166.countries_by_alpha3[i].name for i in iso_codes]

###############################################################################
#create time series for country self-similarity

self_tm = pd.DataFrame(index = iso_codes, columns = years)
for c in self_tm.index:
    c_obs = sorted([cy for cy in meta_mat.index.tolist() if c in cy])
    print c
    for sims in range(1,len(c_obs)):
        if meta_mat.ix[c_obs[sims-1]][c_obs[sims]] != 0:
            self_tm.ix[c][c_obs[sims][-4:]] = meta_mat.ix[c_obs[sims-1]][c_obs[sims]]
        else:
            self_tm.ix[c][c_obs[sims][-4:]] = meta_mat.ix[c_obs[sims]][c_obs[sims-1]]

self_tm.transpose().to_csv(myPath + "doc2vec2\\"  + "country_similarities_tm.csv")

#create time series for similarity with US
us_ind = sorted([c for c in meta_mat.index if 'USA' in c])
us_tm = pd.DataFrame(index = iso_codes, columns = years)
for c in us_tm.index:
    print c
    for sims in range(0,len(us_ind)):
        k = [i for i in meta_mat.index if c in i and us_ind[sims][-4:] in i]
        if len(k) == 1:
            if meta_mat.ix[k[0]][us_ind[sims]] != 0:
                us_tm.ix[c][us_ind[sims][-4:]] = meta_mat.ix[k[0]][us_ind[sims]]
            else:
                us_tm.ix[c][us_ind[sims][-4:]] = meta_mat.ix[us_ind[sims]][k[0]]
            
us_tm.transpose().to_csv(myPath + "doc2vec2\\"  + "us_similarities.csv")

###############################################################################
#check similarity against dyadic data based on voting

dyads = pd.read_csv(myPath + "Dyadicdata.tab", sep = "\t")
