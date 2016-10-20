# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 07:45:47 2016

@author: S
"""

import sys  
reload(sys)  
sys.setdefaultencoding('utf8')

import os 
import numpy as np
import pandas as pd
import igraph as ig
import networkx as nx
import matplotlib.pyplot as plt
plt.style.use('ggplot')

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
countries_tm =  pd.read_csv(myPath + "doc2vec2\\"  + "country_similarities_tm.csv", index_col = 0)


cols_dc = [i + "_degcentr" for i in countries_tm.columns.tolist()]
cols_ec = [i + "_eigcentr" for i in countries_tm.columns.tolist()]
cols_bc = [i + "_betcentr" for i in countries_tm.columns.tolist()]
cols_dc60 = [i + "_degcentr60" for i in countries_tm.columns.tolist()]
cols_ec60 = [i + "_eigcentr60" for i in countries_tm.columns.tolist()]
cols_bc60 = [i + "_betcentr60" for i in countries_tm.columns.tolist()]
cols_nt = ['density', 'density60']

cols = cols_nt + cols_dc + cols_dc60 + cols_ec + cols_ec60 + cols_bc + cols_bc60 
net_df = pd.DataFrame(index = countries_tm.index, columns = cols)


for yy in range(1962, 2015):
    inds = sorted([i for i in meta_mat.index if str(yy) in i])
    if len(inds) > 0:
        meta_mat_yy = meta_mat.ix[inds][inds]

        thresholds = dict()
        for c in meta_mat_yy.index:
            v = np.append(meta_mat_yy.ix[c].values, meta_mat_yy[c].values)
            v = v[v>0]
            thresholds[c] = np.percentile(v, 90)

        meta_mat_yy_adj = meta_mat_yy
        for j in range(0,len(meta_mat_yy_adj.index)):
            for k in range(j+1, len(meta_mat_yy_adj.index)):
                cond1 = meta_mat_yy_adj.ix[meta_mat_yy_adj.index[j]][meta_mat_yy_adj.columns[k]] < thresholds[meta_mat_yy_adj.index[j]]
                cond2 = meta_mat_yy_adj.ix[meta_mat_yy_adj.index[j]][meta_mat_yy_adj.columns[k]] < thresholds[meta_mat_yy_adj.index[k]]
                if cond1 and cond2:
                    meta_mat_yy_adj.ix[meta_mat_yy_adj.index[j]][meta_mat_yy_adj.columns[k]] = 0
        
         
         
        g = ig.Graph.Adjacency((meta_mat_yy_adj.values > 0).tolist())  
        g = g.as_undirected()
        g.es['weight'] = meta_mat_yy_adj.values[meta_mat_yy_adj.values > 0]
        g.vs['name'] = [i[:3] for i in meta_mat_yy_adj[meta_mat_yy_adj > 0].index]
        g.write_gml(myPath + "doc2vec2\\" + str(yy) + '_net.gml')        
        
        net_df.ix[yy]['density'] = g.density()
        dg_cent = g.strength(weights = g.es['weight'])
        ev_cent = g.eigenvector_centrality(directed = False, weights = g.es['weight'])
        bt_cent = g.betweenness(directed = False, weights = g.es['weight'])
        for n in range(0,len(g.vs())):
            try:
                net_df.ix[yy][g.vs()[n]['name'] + "_degcentr"] = dg_cent[n]
                net_df.ix[yy][g.vs()[n]['name'] + "_eigcentr"] = ev_cent[n]
                net_df.ix[yy][g.vs()[n]['name'] + "_betcentr"] = bt_cent[n]
            except ValueError:
                continue
            
            
        g = ig.Graph.Adjacency((meta_mat_yy_adj.values > 0.6).tolist())  
        g = g.as_undirected()
        g.es['weight'] = meta_mat_yy_adj.values[meta_mat_yy_adj.values > 0.6]
        g.vs['name'] = [i[:3] for i in meta_mat_yy_adj[meta_mat_yy_adj > 0.6].index]
        net_df.ix[yy]['density60'] = g.density()
        bt_cent = g.betweenness(directed = False, weights = g.es['weight'])
        ev_cent = g.eigenvector_centrality(directed = False, weights = g.es['weight'])
        bt_cent = g.betweenness(directed = False, weights = g.es['weight'])
        for n in range(0,len(g.vs())):
            try:
                net_df.ix[yy][g.vs()[n]['name'] + "_degcentr60"] = dg_cent[n]
                net_df.ix[yy][g.vs()[n]['name'] + "_eigcentr60"] = ev_cent[n]
                net_df.ix[yy][g.vs()[n]['name'] + "_betcentr60"] = bt_cent[n]
            except ValueError:
                continue
        
net_df.to_csv(myPath + "doc2vec2\\"  + "networks_tm.csv")        







   
        top10 = sorted(range(len(g.degree())), key=lambda i: g.degree()[i])[-10:] 
        
        
        g.es['weight'] = meta_mat_yy_adj.values[meta_mat_yy_adj.values.nonzero()]
        g.es['weight'] = meta_mat_yy_adj.values[meta_mat_yy_adj.values > 0.6].tolist()
        g.vs['label'] = [i[:3] for i in meta_mat_yy_adj.index]

        layout = g.layout("kk")
        ig.plot(g, layout = layout)
         
        g = nx.Graph(meta_mat_yy_adj.values)
        labels = dict()
        for i in g.nodes():
            labels[i] = meta_mat_yy_adj.index[i][:3]
            




        density1.append(nx.density(g))
        degree_sequence = sorted(nx.degree(g).values(),reverse=True)

        e_bunch = [(u,v,d) for (u,v,d) in g.edges(data=True) if d['weight'] < 0.6]
        g.remove_edges_from(e_bunch)
        density2.append(nx.density(g))

pos = nx.spring_layout(g, k = 0.1)
nx.draw_networkx_nodes(g, pos, node_size = 5, node_color = '#000066')
nx.draw_networkx_edges(g, pos, width = 0.5, edge_color='#99CCFF', alpha = 0.5)
#nx.draw_networkx_labels(g,pos,labels,font_size=5)
plt.axis('off')
