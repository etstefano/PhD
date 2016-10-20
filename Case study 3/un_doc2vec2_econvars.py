# -*- coding: utf-8 -*-
"""
Created on Tue May 03 10:35:41 2016

@author: S
"""

import sys  
reload(sys)  
sys.setdefaultencoding('utf8')
from __future__ import print_function


import os 
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('ggplot')

np.random.seed(89)


myPath =  "C:\\Users\\S\\Documents\\Uni\\UNspeeches\\"

data = pd.read_csv(myPath + "doc2vec2\\" + "word2vec_kpca_ip.csv", index_col = 0)
meta_mat = pd.read_csv(myPath + "doc2vec2\\" + "meta_mat.csv", index_col = 0)

###############################################################################
#check relation between idealpoint and US similarity
#
el_us = mat2eg_us(meta_mat)
el_us['year'] = el_us.country1.apply(lambda x: int(x[-4:]))
el_us['iso_code'] = f_isocode(el_us)

el_data = pd.merge(el_us, data,  how='inner', on=['year', 'iso_code'])
plt.scatter(el_data.Idealpoint, el_data.edge)



###############################################################################
#import and merge with econ data

econ = pd.read_csv(myPath + "estimation_dataset.csv") 
#drop years not in data
econ = econ[econ.year.isin(data.year.unique())]
c_codes = pd.read_csv(myPath + "COW country codes.csv")
econ['CountryAbb'] = f_get_abbs(c_codes, econ)

econ_data = pd.merge(econ, data, how = 'inner', on = ['year', 'ccode'])
econ_data.to_csv(myPath + "doc2vec2\\" + "econ_data_merged.csv")


###############################################################################
###############################################################################
def f_get_abbs(c_codes, econ):
    abbs = []
    for i in econ.ccode.values.tolist():
        try:
            abbs.append(c_codes[c_codes.CCode == i].StateAbb.values[0])
        except IndexError:
            abbs.append('NAN')
    return abbs
    
###############################################################################
###############################################################################
def f_isocode(el_us):
    iso_codes = list()
    for i in el_us.index:
        if el_us.ix[i].country1[:3] == 'USA':
            iso_codes.append(el_us.ix[i].country2[:3])
        else:
            iso_codes.append(el_us.ix[i].country1[:3])
    return iso_codes

###############################################################################
###############################################################################
def mat2eg_us(meta_mat):
    country1 = []
    country2 = []
    edge = []
    count = 0
    for i in range(0,len(meta_mat.index)):
        count += 1
        if count%10 == 0:
            print(str(count) + " country-years done out of " + str(len(meta_mat.index)))
            
        for j in range(i+1,len(meta_mat.columns)):
            if ('USA' in meta_mat.index[i] or 'USA' in meta_mat.columns[j]) and meta_mat.index[i][-4:] == meta_mat.columns[j][-4:]:
                country1.append(meta_mat.index[i])
                country2.append(meta_mat.columns[j])
                edge.append(meta_mat.ix[meta_mat.index[i]][meta_mat.columns[j]])
    
    d = {'country1' : country1,
         'country2' : country2,
         'edge': edge}
         
    return  pd.DataFrame(d)