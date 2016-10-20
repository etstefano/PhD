# -*- coding: utf-8 -*-
"""
Created on Tue Apr 05 11:31:26 2016

@author: S
"""

import sys  
reload(sys)  
sys.setdefaultencoding('utf8')

from sklearn.decomposition import PCA, KernelPCA
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import fcluster

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

###############################################################################
myPath = "C:\\Users\\S\\Documents\\Uni\\UNspeeches\\"
#import dataset
data = pd.read_csv(myPath + "doc2vec\\" + "doc2vec_results.csv", index_col = 0)


#get only countries for one year
years_range = range(1962, 2015)

for yy in years_range:
    try:
        index_y = []
        for ix in data.index: 
            try:
                if str(yy) in ix:
                    index_y.append(ix)
            except TypeError:
                continue
        
        data_y = data.ix[index_y]
    
        kpca = KernelPCA(n_components = 3, kernel="cosine", fit_inverse_transform=True, 
                                         gamma=10)
        X_kpca = kpca.fit_transform(data_y.values)
        data_y['kpca_x'] = X_kpca[:,0]
        data_y['kpca_y'] = X_kpca[:,1]
        data_y['kpca_z'] = X_kpca[:,2]
        
        Z = linkage(X_kpca, method = 'ward')
        c, coph_dists = cophenet(Z, pdist(X_kpca))
        k = 2
        data_y['membership'] = fcluster(Z, k, criterion='maxclust')
    
        data_y['iso_code'] = [i[:3] for i in data_y.index.tolist()]
        data_y['names'] = [iso3166.countries_by_alpha3[code].name for code in data_y['iso_code']]
        data_y['year'] = yy
        
        data_y[data_y.columns[-7:]].to_csv(myPath + "doc2vec\\communities_by_kpca\\" 
                                                        + str(yy) + "_kpca.csv")
                                            
                    
        #plot and save                                    
        color_list = plt.cm.Set3(np.linspace(0, float(max(data_y['membership'].values))/10, 
                                             len(data_y['membership'].unique())))
                                             
        colors = [color_list[i-1] for i in data_y['membership'].values]
        
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111, projection='3d')
        ax1.scatter(X_kpca[:,0], X_kpca[:,1], X_kpca[:,2], s = 40,
                                c = colors)
        ax1.set_title("Kernel PCA projection - Year " + str(yy))
        fig1.savefig(myPath + "doc2vec\\communities_by_kpca\\" + str(yy) + "_kpca.png",
                                 dpi = 300)
        
    
        plt.figure(figsize=(25, 15))
        plt.title('Hierarchical Clustering Dendrogram - Year ' + str(yy))
        plt.xlabel('Countries')
        plt.ylabel('Distance')
        dendrogram(
                                Z,
                                leaf_rotation=90.,  # rotates the x axis labels
                                leaf_font_size=12.,  # font size for the x axis labels
                                labels = data_y['names'].values
                                )
        plt.show()
        plt.savefig(myPath + "doc2vec\\communities_by_kpca\\" + str(yy) + "_dendogram.png",
                                dpi = 300)
        plt.close()
        
        del data_y
     
    except ValueError:
        continue           
                
        
              

    
    
#let's try PCA
#pca = PCA(n_components = 3)
#pca.fit(data_y.values)
#X = pca.transform(data_y.values)    
#
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(X[:,0], X[:,1], X[:,2])
###############################################################################
#concatenate all dataframes into one
import os
comms_files = list()
for file in os.listdir(myPath + "doc2vec\\communities_by_kpca\\"):
    if "csv" in file:
        comms_files.append(file)

for cf in comms_files:
    if '1962' in cf:
        df = pd.read_csv(myPath + "doc2vec\\communities_by_kpca\\" + cf, 
                         index_col = 0)
    else:
        df1 = pd.read_csv(myPath + "doc2vec\\communities_by_kpca\\" + cf, 
                         index_col = 0)
                         
        df = pd.concat([df, df1])

df.to_csv(myPath + "doc2vec\\communities_by_kpca\\" + "communities_kpca.csv")