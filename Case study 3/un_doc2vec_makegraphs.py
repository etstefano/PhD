# -*- coding: utf-8 -*-
"""
Created on Fri May 06 10:24:29 2016
This script makes graphs for the paper
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
data = pd.read_csv(myPath + "doc2vec2\\" + "doc2vec_kpca_2.csv", index_col = 0)

aa = data[data.iso_code == 'CHE'][['year', 'env_similarity']].sort(columns = 'year')
bb = data[data.iso_code == 'USA'][['year', 'env_similarity']].sort(columns = 'year')
 
years = sorted(data.year.unique())

b = data[['env_similarity', 'year']].groupby('year').mean()
c = data[['ter_similarity', 'year']].groupby('year').mean()

env_scores = pd.read_csv(myPath + "2014_env_scores.csv")

env = pd.merge(data[data.year == 2014][['iso_code', 'env_similarity']], env_scores, 
               how = 'inner', on = ['iso_code'])
               
###############################################################################
#time series for islamic terror index
               
c['ter_index'] = c['ter_similarity'].apply(lambda x: (x-c.ix[1995]['ter_similarity'])/c.ix[1995]['ter_similarity'])
               
               
fig_terror = plt.figure()
ax_terror = fig_terror.add_subplot(111)
ax_terror.plot(c.index[1:], c.ter_index[1:],
               color = 'k')
ax_terror.plot(c.index[1:], np.zeros(len(c.index[1:])), 
               linewidth = 1.5, 
               linestyle = "--",
               color = '0.65')
ax_terror.set_xticks(c.index[1:])
ax_terror.set_xticklabels(c.index[1:], rotation = 'vertical', fontsize = 14)
ax_terror.set_xlabel('Year', fontsize = 16)
ax_terror.set_ylabel('Terror Index', fontsize = 16)
ax_terror.set_ylim(-3,3)
###############################################################################
#kpca versus idealpoint

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import DBSCAN
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import fcluster
kpca_ip = pd.read_csv(myPath + 'doc2vec2\\' + 'word2vec_kpca_ip.csv', index_col = 0)

X = kpca_ip[kpca_ip.year == 1975][['kpca_x', 'Idealpoint']].values
db = DBSCAN(eps = 0.2).fit(X)
labels = db.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)


range_n_clusters = [2, 3, 4, 5, 6]
cl_ts = pd.DataFrame(index = sorted(data.year.unique()), 
                     columns = [str(i) for i in range_n_clusters])
for n_clusters in range_n_clusters:
    for y in cl_ts.index:
        X = kpca_ip[kpca_ip.year == y][['kpca_x', 'Idealpoint']].values
        clusterer = KMeans(n_clusters = n_clusters, random_state = 10)
        cluster_labels = clusterer.fit_predict(X)
        cl_ts.ix[y][str(n_clusters)] = silhouette_score(X, cluster_labels)
        
for n_clusters in range_n_clusters:
    for y in cl_ts.index:
        X = kpca_ip[kpca_ip.year == y][['kpca_x', 'Idealpoint']].values
        Z = linkage(X, method = 'ward')
        k = n_clusters
        cluster_labels = fcluster(Z, k, criterion='maxclust')
        cl_ts.ix[y][str(n_clusters)] = silhouette_score(X, cluster_labels)


clusterer = KMeans(n_clusters = 4, random_state = 10)
cl1 = clusterer.fit_predict(kpca_ip[kpca_ip.year == 1980][['kpca_x', 'Idealpoint']].values)
cl2 = clusterer.fit_predict(kpca_ip[kpca_ip.year == 1990][['kpca_x', 'Idealpoint']].values)
cl3 = clusterer.fit_predict(kpca_ip[kpca_ip.year == 2000][['kpca_x', 'Idealpoint']].values)
cl4 = clusterer.fit_predict(kpca_ip[kpca_ip.year == 2014][['kpca_x', 'Idealpoint']].values)

color_list = plt.cm.gray(np.linspace(0, float(max(cl1))/10, 4))

fig_kpca = plt.figure()
ax_kpca1 = fig_kpca.add_subplot(221)
colors1 = [color_list[i-1] for i in cl1]
ax_kpca1.scatter(kpca_ip[kpca_ip.year == 1980].Idealpoint, 
                 kpca_ip[kpca_ip.year == 1980].kpca_x,
                 s = 30, c = 'k')
ax_kpca1.set_ylabel('Kernel PCA - 1st Coordinate', fontsize = 13)
ax_kpca1.set_xlabel('Idealpoint', fontsize = 13)
ax_kpca1.text(0.05, 0.95, '1980', transform=ax_kpca1.transAxes, fontsize=12,
        verticalalignment='top')
ax_kpca1.set_xlim(-3,3)
ax_kpca1.set_ylim(-0.8,0.8)


ax_kpca2 = fig_kpca.add_subplot(222)
ax_kpca2.scatter(kpca_ip[kpca_ip.year == 1990].Idealpoint, 
                 kpca_ip[kpca_ip.year == 1990].kpca_x,
                    s = 30, c = 'k')
ax_kpca2.set_xlabel('Idealpoint', fontsize = 13)
ax_kpca2.text(0.05, 0.95, '1990', transform=ax_kpca2.transAxes, fontsize=12,
        verticalalignment='top')
ax_kpca2.set_xlim(-3,3)
ax_kpca2.set_ylim(-0.8,0.8)

ax_kpca3 = fig_kpca.add_subplot(223)
ax_kpca3.scatter(kpca_ip[kpca_ip.year == 2000].Idealpoint, 
                 kpca_ip[kpca_ip.year == 2000].kpca_x,
                    s = 30, c = 'k')
ax_kpca3.set_ylabel('Kernel PCA - 1st Coordinate', fontsize = 13)
ax_kpca3.set_xlabel('Idealpoint', fontsize = 13)
ax_kpca3.text(0.05, 0.95, '2000', transform=ax_kpca3.transAxes, fontsize=12,
        verticalalignment='top')
ax_kpca3.set_xlim(-3,3)
ax_kpca3.set_ylim(-0.8,0.8)

ax_kpca4 = fig_kpca.add_subplot(224)
ax_kpca4.scatter(kpca_ip[kpca_ip.year == 2014].Idealpoint, 
                 kpca_ip[kpca_ip.year == 2014].kpca_x,
                    s = 30, c = 'k')
ax_kpca4.set_xlabel('Idealpoint', fontsize = 13,)
ax_kpca4.text(0.05, 0.95, '2014', transform=ax_kpca4.transAxes, fontsize=12,
        verticalalignment='top')
ax_kpca4.set_xlim(-3,3)
ax_kpca4.set_ylim(-0.8,0.8)

###############################################################################
###############################################################################
net_tm = pd.read_csv(myPath + "doc2vec2\\" + "networks_tm.csv", index_col = 0)
net_tm['density_index'] = net_tm.density.apply(lambda x: 
                            (x-net_tm.ix[1970].density)/net_tm.ix[1970].density)
net_tm['density60_index'] = net_tm.density60.apply(lambda x: 
                            (x-net_tm.ix[1970].density60)/net_tm.ix[1970].density60)

fig_density = plt.figure()
ax_density = fig_density.add_subplot(111)
line1, = ax_density.plot(net_tm.index[1:], net_tm.density_index[1:], linestyle = '--', color = '0.10')
line2, = ax_density.plot(net_tm.index[1:], net_tm.density60_index[1:], linestyle = '-', color = '0')
ax_density.legend((line1, line2), ('Semantic network - Unfiltered', 'Semantic network - Filtered'))
ax_density.set_ylabel('Density', fontsize = 18)
ax_density.set_xlabel('Year', fontsize = 18)

#Russia and USA
net_tm['RUS_eigcentr60_index'] = net_tm.RUS_eigcentr60.apply(lambda x:
                                    (x-net_tm.ix[1971].RUS_eigcentr60)/net_tm.ix[1971].RUS_eigcentr60)
net_tm['USA_eigcentr60_index'] = net_tm.USA_eigcentr60.apply(lambda x:
                                    (x-net_tm.ix[1971].USA_eigcentr60)/net_tm.ix[1971].USA_eigcentr60)

fig_russia = plt.figure()
ax_russia = fig_russia.add_subplot(111)
line1, = ax_russia.plot(net_tm.index[2:], net_tm.USA_eigcentr60_index[2:], linestyle = '--', color = '0.10')
line2, = ax_russia.plot(net_tm.index[2:], net_tm.RUS_eigcentr60_index[2:], linestyle = '-', color = '0')
ax_russia.set_ylim(-2,2)
ax_russia.legend((line1, line2), ('United States of America', 'USSR/Russian Federation'))
ax_russia.set_ylabel('Centrality Index', fontsize = 18)
ax_russia.set_xlabel('Year', fontsize = 18)

#EU15 vs Brics
eu15 = ['AUT', 'BEL', 'DNK', 'FIN', 'FRA', 'DEU', 'GRC', 'IRL', 'ITA', 'LUX', 
      'NLD', 'PRT', 'ESP', 'SWE', 'BGR']
brics = ['BRA', 'IND', 'RUS', 'CHN', 'ZAF', 'KOR', 'MEX', 'IDN', 'TUR', 'SAU']

eu15_ec = [i+"_eigcentr60" for i in eu15]
brics_ec = [i + "_eigcentr60" for i in brics]
net_tm['eu15_ec'] = net_tm[eu15_ec].mean(axis = 1)
net_tm['brics_ec'] = net_tm[brics_ec].mean(axis = 1)

ec_cols = [i for i in net_tm.columns if '_eigcentr60' in i and 'index' not in i]
net_tm['mean_ec60'] = net_tm[ec_cols].mean(axis = 1)


eu15_ec_index = [(net_tm.ix[i].eu15_ec - net_tm.ix[i].mean_ec60)/net_tm.ix[i].mean_ec60 for 
                    i in net_tm.index]
brics_ec_index = [(net_tm.ix[i].brics_ec - net_tm.ix[i].mean_ec60)/net_tm.ix[i].mean_ec60 for 
                    i in net_tm.index]
net_tm['eu15_ec_index'] = eu15_ec_index
net_tm['brics_ec_index'] =brics_ec_index

eu_bric_diff = [net_tm.ix[i].eu15_ec_index - net_tm.ix[i].brics_ec_index for 
                    i in net_tm.index]
net_tm['eu15_brics_diff'] = eu_bric_diff



fig_eb = plt.figure()
ax_eb = fig_eb.add_subplot(211)
line1, = ax_eb.plot(net_tm.index[31:], net_tm.eu15_ec_index[31:], linestyle = '--', color = '0.10')
line2, = ax_eb.plot(net_tm.index[31:], net_tm.brics_ec_index[31:], linestyle = '-', color = '0')
ax_eb.set_ylim(-0.8, 0.8)
ax_eb.set_ylabel('Centrality Index', fontsize = 15)
ax_eb.legend((line1, line2), ('EU 15', 'Emerging economies'))


ax_eb_d = fig_eb.add_subplot(212)
ax_eb_d.fill_between(net_tm.index[31:], 0, net_tm['eu15_brics_diff'][31:])
ax_eb_d.set_ylim(-1, 1)
ax_eb_d.set_xlabel('Year', fontsize = 18)
ax_eb_d.set_ylabel('Difference in centrality index', fontsize = 15)


###############################################################################
#time series for education index
edu = data[['edu_similarity', 'year']].groupby('year').mean()
               
edu['edu_index'] = edu['edu_similarity'].apply(lambda x: (x-edu.ix[1995]['edu_similarity'])/edu.ix[1995]['edu_similarity'])
               
               
fig_edu = plt.figure(figsize = (11,5))
ax_edu = fig_edu.add_subplot(111)
ax_edu.plot(edu.index[1:], edu.edu_index[1:],
               color = 'k')
ax_edu.plot(edu.index[1:], np.zeros(len(edu.index[1:])), 
               linewidth = 1.5, 
               linestyle = "--",
               color = '0.65')
#ax_edu.set_xticks(edu.index[1:])
#ax_edu.set_xticklabels(edu.index[1:], rotation = 'vertical', fontsize = 14)
ax_edu.set_xlabel('Year', fontsize = 16)
ax_edu.set_ylabel('Education Index', fontsize = 16)
ax_edu.set_ylim(-3,3)

###############################################################################
#times series for health index
hea = data[['hea_similarity', 'year']].groupby('year').mean()
               
hea['hea_index'] = hea['hea_similarity'].apply(lambda x: (x-hea.ix[1995]['hea_similarity'])/np.abs(hea.ix[1995]['hea_similarity']))
               
               
fig_hea = plt.figure(figsize = (11,5))
ax_hea = fig_hea.add_subplot(111)
ax_hea.plot(hea.index[1:], hea.hea_index[1:],
               color = 'k')
ax_hea.plot(hea.index[1:], np.zeros(len(hea.index[1:])), 
               linewidth = 1.5, 
               linestyle = "--",
               color = '0.65')
#ax_edu.set_xticks(edu.index[1:])
#ax_edu.set_xticklabels(edu.index[1:], rotation = 'vertical', fontsize = 14)
ax_hea.set_xlabel('Year', fontsize = 16)
ax_hea.set_ylabel('Health Index', fontsize = 16)
ax_hea.set_ylim(-30,30)

###############################################################################
#time series for nuclear index
nuc = data[['nuc_similarity', 'year']].groupby('year').mean()
               
nuc['nuc_index'] = nuc['nuc_similarity'].apply(lambda x: 
    (x-nuc.ix[1995]['nuc_similarity'])/np.abs(nuc.ix[1995]['nuc_similarity']))
               
               
fig_nuc = plt.figure(figsize = (11,5))
ax_nuc = fig_nuc.add_subplot(111)
ax_nuc.plot(nuc.index[1:], nuc.nuc_index[1:],
               color = 'k')
ax_nuc.plot(nuc.index[1:], np.zeros(len(nuc.index[1:])), 
               linewidth = 1.5, 
               linestyle = "--",
               color = '0.65')
#ax_edu.set_xticks(edu.index[1:])
#ax_edu.set_xticklabels(edu.index[1:], rotation = 'vertical', fontsize = 14)
ax_nuc.set_xlabel('Year', fontsize = 16)
ax_nuc.set_ylabel('Nuclear weapon Index', fontsize = 16)
ax_nuc.set_ylim(-3,3)

###############################################################################
#time series for terror index
ter = data[['ter_similarity', 'year']].groupby('year').mean()
               
ter['ter_index'] = ter['ter_similarity'].apply(lambda x: 
    (x-ter.ix[1995]['ter_similarity'])/np.abs(ter.ix[1995]['ter_similarity']))
               
               
fig_ter = plt.figure(figsize = (11,5))
ax_ter = fig_ter.add_subplot(111)
ax_ter.plot(ter.index[10:], ter.ter_index[10:],
               color = 'k')
ax_ter.plot(ter.index[10:], np.zeros(len(ter.index[10:])), 
               linewidth = 1.5, 
               linestyle = "--",
               color = '0.65')
#ax_edu.set_xticks(edu.index[1:])
#ax_edu.set_xticklabels(edu.index[1:], rotation = 'vertical', fontsize = 14)
ax_ter.set_xlabel('Year', fontsize = 16)
ax_ter.set_ylabel('Islamic terror Index', fontsize = 16)
ax_ter.set_ylim(-3,3)


###############################################################################
#centrality indices
net_tm = pd.read_csv(myPath + "doc2vec2\\" + "networks_tm.csv", index_col = 0)

cols_ecRUS = [i for i in net_tm.columns if 'eigcentr60' in i]


net_tm['eigcentr_mean'] = net_tm[cols_ecRUS].mean(axis = 1)
rus = [(net_tm.ix[i].RUS_eigcentr60 - net_tm.ix[i].eigcentr_mean)/
        net_tm.ix[i].eigcentr_mean for i in net_tm.index]
net_tm['RUS_eigcentr60_index'] = rus

usa = [(net_tm.ix[i].USA_eigcentr60 - net_tm.ix[i].eigcentr_mean)/
        net_tm.ix[i].eigcentr_mean for i in net_tm.index]
net_tm['USA_eigcentr60_index'] = usa        

chn = [(net_tm.ix[i].CHN_eigcentr60 - net_tm.ix[i].eigcentr_mean)/
        net_tm.ix[i].eigcentr_mean for i in net_tm.index]
net_tm['CHN_eigcentr60_index'] = chn        

net_tm['RUS_eigcentr60_index_chg'] = net_tm['RUS_eigcentr60_index'].apply(lambda x: 
    (x-net_tm.ix[1995]['RUS_eigcentr60_index'])/np.abs(net_tm.ix[1995]['RUS_eigcentr60_index']))
net_tm['USA_eigcentr60_index_chg'] = net_tm['USA_eigcentr60_index'].apply(lambda x: 
    (x-net_tm.ix[1995]['USA_eigcentr60_index'])/np.abs(net_tm.ix[1995]['USA_eigcentr60_index']))
net_tm['CHN_eigcentr60_index_chg'] = net_tm['CHN_eigcentr60_index'].apply(lambda x: 
    (x-net_tm.ix[1995]['CHN_eigcentr60_index'])/np.abs(net_tm.ix[1995]['CHN_eigcentr60_index']))

fig_countries = plt.figure(figsize = (11,5))
ax_cc = fig_countries.add_subplot(111)
line1, = ax_cc.plot(net_tm.index[2:], net_tm['RUS_eigcentr60_index_chg'][2:],
                    linestyle = "--",
                    color = '0.20')
line2, = ax_cc.plot(net_tm.index[2:], net_tm['USA_eigcentr60_index_chg'][2:],
                    linestyle = "-",
                    color = '0')
#line3, = ax_cc.plot(net_tm.index[2:], net_tm['CHN_eigcentr60_index_chg'][2:])
ax_cc.legend((line1, line2), ('USSR/Russian Federation', 'USA'), loc = 2)
ax_cc.set_ylabel('Centrality Index', fontsize = 14)
ax_cc.set_xlabel('Year', fontsize = 14) 
ax_cc.set_ylim(-1,4)

###############################################################################
#image for country numbers and avg no of tokens
avg = data.groupby('year').count()['0']
tokens = pd.read_csv(myPath + 'tokens.csv', index_col = 0 )

fig, ax1 = plt.subplots()
ax1.plot(avg.index[1:], avg.values[1:], color = '0.1')
ax1.set_xlabel('Year', fontsize = 14)
ax1.set_ylabel('Number of nations', fontsize = 14)
ax1.grid('off')
ax2 = ax1.twinx()
ax2.plot(avg.index[1:], tokens.tokens, linestyle = '--', color = "0")
ax2.set_ylabel('Tokens - mean frequency', fontsize = 14)
