# -*- coding: utf-8 -*-
"""
Created on Mon May 02 13:21:01 2016
This script builds a time series of cosing similarity scores based on outcomes
from word2vec analysis of UN speeches
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

#transform into edgelist
el = mat2eg(meta_mat)
el.to_csv(myPath + "doc2vec2\\" + "edge_list.csv")
#drop rows with 1962
cond = ['1962' not in el.ix[i]['country1'] or '1962' not in el.ix[i]['country2'] for i in el.index]
el = el[cond]

###############################################################################
###############################################################################
def mat2eg(meta_mat):
    country1 = []
    country2 = []
    edge = []
    count = 0
    for i in range(0,len(meta_mat.index)):
        count += 1
        if count%10 == 0:
            print str(count) + " country-years done out of " + str(len(meta_mat.index))
            
        for j in range(i+1,len(meta_mat.columns)):
            country1.append(meta_mat.index[i])
            country2.append(meta_mat.columns[j])
            edge.append(meta_mat.ix[meta_mat.index[i]][meta_mat.columns[j]])
    
    d = {'country1' : country1,
         'country2' : country2,
         'edge': edge}
         
    return  pd.DataFrame(d)