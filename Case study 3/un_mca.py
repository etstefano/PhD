# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 18:19:03 2016

This script performs Multi Correspondence Analysis on UN Voting data

@author: S
"""

import sys  
reload(sys)  
sys.setdefaultencoding('utf8')

import os
import numpy as np
import pandas as pd
import mca

from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import fcluster

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

myPath = "C:\\Users\\S\\Documents\\Uni\\UNspeeches\\"

#import list of files
un_voting_files = list()
for file in os.listdir(myPath + "un_voting\\"):
    if 'piv' in file:
        un_voting_files.append(file)
        
#each file is one year. For each file/year, import the voting matrix
#and perform mca. Then export results
for f in un_voting_files[1:]:
    yy = f[:4]
    #import file
    un_votes = pd.read_csv(myPath + "un_voting\\" + f, index_col = 0)
    #get rid of rows with inactive states
    un_votes = un_votes[un_votes[un_votes.columns[0]]!= 9]
    
    #perform mca
    mca_model = mca.MCA(un_votes[un_votes.columns[:-2]])
    coords = mca_model.fs_r(N=3)
    
    #assing values to dataframe
    un_votes['mca_x'] = coords[:,0]
    un_votes['mca_y'] = coords[:,1]
    un_votes['mca_z'] = coords[:,2]
    
    #perform clustering using ward method, and assign membership
    Z = linkage(coords, method = 'ward')
    c, coph_dists = cophenet(Z, pdist(coords))
    k = 2
    un_votes['membership'] = fcluster(Z, k, criterion='maxclust')
    un_votes.to_csv(myPath + "un_voting\\communities_by_mca\\" + f)
    

    #plot first 3 coordinates
    color_list = plt.cm.Set3(np.linspace(0, float(max(un_votes['membership'].values))/10, 
                                             len(un_votes['membership'].unique())))
                                             
    colors = [color_list[i-1] for i in un_votes['membership'].values]   
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.scatter(coords[:,0], coords[:,1], coords[:,2], s = 40,
                                c = colors)
    ax1.set_title("MCA projection - Year " + str(yy))
    fig1.savefig(myPath + "un_voting\\communities_by_mca\\" + str(yy) + "_voting_mca.png",
                                 dpi = 300)
    
    #plot dendogram
    plt.figure(figsize=(25, 15))
    plt.title('Hierarchical Clustering Dendrogram, MCA - Year ' + yy)
    plt.xlabel('Countries')
    plt.ylabel('Distance')
    dendrogram(
                            Z,
                            leaf_rotation=90.,  # rotates the x axis labels
                            leaf_font_size=12.,  # font size for the x axis labels
                            labels = un_votes['iso_code'].values
                            )
    plt.show()
    plt.savefig(myPath + "un_voting\\communities_by_mca\\" + str(yy) + "_voting_dendogram.png",
                                dpi = 300)
    plt.close()