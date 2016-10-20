# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 13:11:44 2016
This script performs regression analysis between PCA results on deep learning
model, and ideal points estimates coming from UN voting data. To test if there
is any predictive power.

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

from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics.pairwise import cosine_similarity

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('ggplot')


import iso3166 
plt.style.use('ggplot')

np.random.seed(89)


myPath = "C:\\Users\\S\\Documents\\Uni\\UNspeeches\\"

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
    

#import ideal points dataset
ip = pd.read_csv(myPath + "Idealpoints.tab", sep = "\t")

#import doc2vec results
doc2vec = pd.read_csv(myPath + "doc2vec2\\" + "doc2vec_kpca.csv", 
                      index_col = 0)

#add columns to doc2vec results
#select only countries
#inds = []
#for i in doc2vec.index:
#    try:
#        inds.append(i[0].isupper())
#    except TypeError:
#        inds.append(False)
#        continue
#doc2vec = doc2vec[inds]
#doc2vec['year'] = [int(i[-4:]) for i in doc2vec.index.values]
#doc2vec['iso_code'] = [i[:3] for i in doc2vec.index.values]
#doc2vec['name'] = [country_dict[i].name for i in doc2vec.iso_code.values]

#filter ip only for years in kpca
ip = ip[ip.Year.isin(doc2vec.year.unique())]

###############################################################################
#match with year and country abbreviation 

#let's first deal with country abbreviations
abbrevs = ip.CountryAbb.values.tolist()
abbrevs = f_match_abbrevs(abbrevs, country_dict)
ip['iso_code'] = abbrevs

#join the two datasets on year and iso_code
ip_newcols = ['year',
 'ccode',
 'CountryAbb',
 'session',
 'N_full',
 'CountryName',
 'CountrySession',
 'Idealpoint',
 'Thetamin',
 'Theta5thpct',
 'Theta10thpct',
 'Theta50thpct',
 'Theta90thpct',
 'Theta95thpct',
 'Thetamax',
 'PctAgreeUS',
 'PctAgreeRUSSIA',
 'PctAgreeBrazil',
 'PctAgreeChina',
 'PctAgreeIndia',
 'PctAgreeIsrael',
 'yObs1',
 'yObs2',
 'yObs3',
 'Nimportant',
 'IdealImportant',
 'ThetaminImp',
 'Theta5thpctImp',
 'Theta10thpctImp',
 'Theta50thpctImp',
 'Theta90thpctImp',
 'Theta95thpctImp',
 'ThetamaxImp',
 'PctAgreeUSImp',
 'iso_code']
 
ip.columns = ip_newcols
ip_kpca = pd.merge(ip, doc2vec, how='inner', on=['year', 'iso_code'])
ip_kpca.to_csv(myPath + "doc2vec2\\" + "word2vec_kpca_ip.csv")

color_list = plt.cm.Set3(np.linspace(0, float(max(ip_kpca['membership'].values))/10, 
                                     len(ip_kpca['membership'].unique())))
                                     
colors = [color_list[i-1] for i in ip_kpca['membership'].values]

fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')
ax1.scatter(ip_kpca[ip_kpca.year == 2014].Idealpoint.values, ip_kpca[ip_kpca.year == 2014].kpca_x.values, 
            ip_kpca[ip_kpca.year == 2014].kpca_y.values, s = 40, c = colors)
ax1.set_title("Ideal Point vs Kernel PCA Coordinates")
ax1.set_xlabel('Ideal point')
ax1.set_ylabel('Kernel PCA, Coordinate 1')
ax1.set_zlabel('Kernel PCA, Coordinate 2')

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.scatter(ip_kpca[ip_kpca.year == 2014].Idealpoint.values, ip_kpca[ip_kpca.year == 2014].kpca_x.values, 
            s = 40, c = colors)
ax2.set_title("Ideal Point vs Kernel PCA Coordinates")
ax2.set_xlabel('Ideal point')
ax2.set_ylabel('Kernel PCA, Coordinate 1')


fig1.savefig(myPath + "doc2vec2\\" + str(yy) + "_kpca.png",
                         dpi = 300)

###############################################################################
#select a machine learning model for PctAgreeUS
ip_kpca_us = ip_kpca[ip_kpca.PctAgreeUS.notnull()]

#obtain validation/test samples
X_train, X_test, y_train, y_test = train_test_split(
    ip_kpca[ip_kpca.columns[35:-5]].values, 
    ip_kpca.PctAgreeUS.values, test_size=0.2, random_state=0)

#neural net
parameters_n_us = {'hidden_layer_sizes': [(600,2), (600,3), (600, 4), (600, 5),
                                          (600,6)], 
              'activation': ['logistic'], 
              'max_iter': [400, 500, 600]}
model_n_us = MLPRegressor()
clf_n_us = GridSearchCV(model_n_us, parameters_n_us, verbose = 2, scoring = 'mean_squared_error')
clf_n_us.fit(X_train, y_train)

best_params = clf_n_us.best_params_
y_pred_n_us = clf_n_us.predict(X_test)
error = mean_squared_error(y_test, y_pred_n_us)

#random forests
parameters_r_us = {'n_estimators': [200, 400, 600]}
model_r_us = RandomForestRegressor()
clf_r_us = GridSearchCV(model_r_us, parameters_r_us, verbose = 2, scoring = 'mean_squared_error')
clf_r_us.fit(X_train, y_train)

y_pred_r_us = clf_r_us.predict(X_test)
error_r_us = mean_squared_error(y_test, y_pred_r_us)

#knn regressor
def cosine(x,y):
    return cosine_similarity(x,y)

params_knn = {'n_neighbors':[40, 50, 60], 'weights': ['uniform', 'distance'],
              'metric': ['euclidean', 'minkowski', 'manhattan']}
               
model_knn_us = KNeighborsRegressor()
clf_knn_us = GridSearchCV(model_knn_us, params_knn, verbose = 2, 
                          scoring = 'mean_squared_error')
clf_knn_us.fit(X_train, y_train)

y_pred_knn_us = clf_knn_us.predict(X_test)
error_knn_us = mean_squared_error(y_test, y_pred_knn_us)

###############################################################################
#selecting a multi-layer neural net model to test if speeches data has predictive
#on voting behaviour

#obtain validation/test samples
X_train, X_test, y_train, y_test = train_test_split(
    ip_kpca[ip_kpca.columns[35:-1]].values, 
    ip_kpca.Idealpoint.values, test_size=0.2, random_state=0)

#set parameters to play with
parameters = {'hidden_layer_sizes': [(100,2), (100,3), (100, 4), (100,5), 
              (100,6)], 'activation': ['logistic','tanh'], 
              'max_iter': [200, 250, 300]}
n_model = MLPRegressor()
clf = GridSearchCV(n_model, parameters, verbose = 2, scoring = 'mean_squared_error')
clf.fit(X_train, y_train)

best_params = clf.best_params_
y_pred = clf.predict(X_test)
error = mean_squared_error(y_test, y_pred)

#use random forest regression
parameters_rf = {'n_estimators': [50, 75, 100, 125, 150, 175]}
r_model = RandomForestRegressor()
clf_r = GridSearchCV(r_model, parameters_rf, verbose = 2, scoring = 'mean_squared_error')
clf_r.fit(X_train, y_train)

y_pred_r = clf_r.predict(X_test)
error_r = mean_squared_error(y_test, y_pred_r)


#let's try support vector machines on doc2vec features
params_svm = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 
              'epsilon': [0.05, 0.1, 0.15, 0.2]}
svm_model = SVR()
clf_svm = GridSearchCV(svm_model, params_svm, verbose = 2, scoring = 'mean_squared_error')
clf_svm.fit(X_train, y_train)

y_pred_svm = clf_svm.predict(X_test)
error_svm = mean_squared_error(y_test, y_pred_svm)

#neural net
X2_train, X2_test, y2_train, y2_test = train_test_split(
    ip_kpca[[u'kpca_x', u'kpca_y', u'kpca_z']].values, 
    ip_kpca.Idealpoint.values, test_size=0.2, random_state=0)

#set parameters to play with
parameters_npca = {'hidden_layer_sizes': [(100,2), (100,3), (100, 4), (100,5), 
              (100,6)], 'activation': ['logistic','tanh'], 
              'max_iter': [200, 250, 300]}
n_model_pca = MLPRegressor()
clf_npca = GridSearchCV(n_model_pca, parameters_npca, verbose = 2, scoring = 'mean_squared_error')
clf_npca.fit(X2_train, y2_train)

best_params = clf_npca.best_params_
y2_pred_n = clf_npca.predict(X2_test)
error_npca = mean_squared_error(y2_test, y2_pred_n)

#random forest
parameters_rf_pca = {'n_estimators': [300, 400, 500]}
r_model_pca = RandomForestRegressor()
clf_r_pca = GridSearchCV(r_model_pca, parameters_rf_pca, verbose = 2, scoring = 'mean_squared_error')
clf_r_pca.fit(X2_train, y2_train)

y_pred_r_pca = clf_r_pca.predict(X2_test)
error_r_pca = mean_squared_error(y2_test, y_pred_r_pca)
###############################################################################
#run OLS regression on 4 vars using kPCA coordinates as predictors: 
#PctAgreeUS  Idealpoint IdealImportant PctAgreeUSImp

#Idealpoint
model_Idealpoint = sm.OLS(ip_kpca.Idealpoint.values, 
                          ip_kpca[[u'kpca_x', u'kpca_y', u'kpca_z']])
results_Idealpoint = model_Idealpoint.fit()
print(results_Idealpoint.summary())

#Idealimportant
ip_kpca_temp = ip_kpca[ip_kpca.IdealImportant.notnull()]
model_IdealImportant = sm.OLS(ip_kpca_temp.IdealImportant.values, 
                          ip_kpca_temp[[u'kpca_x', u'kpca_y', u'kpca_z']])
results_IdealImportant = model_IdealImportant.fit()
print(results_IdealImportant.summary())

#PctAgreeUS
ip_kpca_temp = ip_kpca[ip_kpca.PctAgreeUS.notnull()]
model_PctAgreeUS = sm.OLS(ip_kpca_temp.PctAgreeUS.values, 
                          ip_kpca_temp[[u'kpca_x', u'kpca_y', u'kpca_z']])
results_PctAgreeUS = model_PctAgreeUS.fit()
print(results_PctAgreeUS.summary())

#PctAgreeUSImp
ip_kpca_temp = ip_kpca[ip_kpca.PctAgreeUSImp.notnull()]
model_PctAgreeUSImp = sm.OLS(ip_kpca_temp.PctAgreeUSImp.values, 
                          ip_kpca_temp[[u'kpca_x', u'kpca_y', u'kpca_z']])
results_PctAgreeUSImpS = model_PctAgreeUSImp.fit()
print(results_PctAgreeUSImpS.summary())

#PctAgreeUS  Idealpoint IdealImportant PctAgreeUSImp

###############################################################################
#function matching abbreviations with iso_codes alpha3
def f_match_abbrevs(abbrevs, country_dict):
    abbrevs_adj = list()
    iso_codes = country_dict.keys()
    for a in abbrevs:
        if a in iso_codes:
            abbrevs_adj.append(a)
        elif a == 'HAI':
             abbrevs_adj.append('HTI')   
        elif a == 'TRI':
             abbrevs_adj.append('TTO') 
        elif a == 'GUA':
             abbrevs_adj.append('GTM') 
        elif a == 'HON':
             abbrevs_adj.append('HND') 
        elif a == 'SAL':
             abbrevs_adj.append('SLV') 
        elif a == 'COS':
             abbrevs_adj.append('CRI') 
        elif a == 'PAR':
             abbrevs_adj.append('PRY') 
        elif a == 'URU':
             abbrevs_adj.append('URY') 
        elif a == 'UKG':
             abbrevs_adj.append('GBR') 
        elif a == 'IRE':
             abbrevs_adj.append('IRL') 
        elif a == 'NTH':
             abbrevs_adj.append('NLD') 
        elif a == 'FRN':
             abbrevs_adj.append('FRA') 
        elif a == 'SPN':
             abbrevs_adj.append('ESP') 
        elif a == 'POR':
             abbrevs_adj.append('PRT') 
        elif a == 'BUL':
             abbrevs_adj.append('BGR')
        elif a == 'ROM':
             abbrevs_adj.append('ROU') 
        elif a == 'SWD':
             abbrevs_adj.append('SWE') 
        elif a == 'DEN':
             abbrevs_adj.append('DNK') 
        elif a == 'ICE':
             abbrevs_adj.append('ISL') 
        elif a == 'MAA':
             abbrevs_adj.append('MRT') 
        elif a == 'NIR':
             abbrevs_adj.append('NER') 
        elif a == 'CDI':
             abbrevs_adj.append('CIV') 
        elif a == 'GUI':
             abbrevs_adj.append('GIN') 
        elif a == 'PAL':
             abbrevs_adj.append('PLW') 
        elif a == 'MSI':
             abbrevs_adj.append('MHL') 
        elif a == 'NAU':
             abbrevs_adj.append('NRU') 
        elif a == 'FIJ':
             abbrevs_adj.append('FJI') 
        elif a == 'SOL':
             abbrevs_adj.append('SLB') 
        elif a == 'VAN':
             abbrevs_adj.append('VUT') 
        elif a == 'NEW':
             abbrevs_adj.append('NZL') 
        elif a == 'AUL':
             abbrevs_adj.append('AUS') 
        elif a == 'ETM':
             abbrevs_adj.append('TLS') 
        elif a == 'INS':
             abbrevs_adj.append('IDN') 
        elif a == 'PHI':
             abbrevs_adj.append('PHL') 
        elif a == 'BRU':
             abbrevs_adj.append('BRN') 
        elif a == 'SIN':
             abbrevs_adj.append('SGP') 
        elif a == 'MAL':
             abbrevs_adj.append('MYS') 
        elif a == 'DRV':
             abbrevs_adj.append('VNM') 
        elif a == 'CAM':
             abbrevs_adj.append('KHM') 
        elif a == 'THI':
             abbrevs_adj.append('THA') 
        elif a == 'NEP':
             abbrevs_adj.append('NPL') 
        elif a == 'MAD':
             abbrevs_adj.append('MDV') 
        elif a == 'GUA':
             abbrevs_adj.append('GTM') 
        elif a == 'SRI':
             abbrevs_adj.append('LKA') 
        elif a == 'MYA':
             abbrevs_adj.append('MMR') 
        elif a == 'BNG':
             abbrevs_adj.append('BGD') 
        elif a == 'BHU':
             abbrevs_adj.append('BTN') 
        elif a == 'ROK':
             abbrevs_adj.append('PRK') 
        elif a == 'MON':
             abbrevs_adj.append('MNG') 
        elif a == 'KZK':
             abbrevs_adj.append('KAZ') 
        elif a == 'KYR':
             abbrevs_adj.append('KGZ') 
        elif a == 'TAJ':
             abbrevs_adj.append('TJK') 
        elif a == 'OMA':
             abbrevs_adj.append('OMN') 
        elif a == 'UAE':
             abbrevs_adj.append('ARE') 
        elif a == 'BAH':
             abbrevs_adj.append('BHR') 
        elif a == 'KUW':
             abbrevs_adj.append('KWT')    
        elif a == 'YAR':
             abbrevs_adj.append('YEM') 
        elif a == 'LEB':
             abbrevs_adj.append('LBN') 
        elif a == 'SSUD':
             abbrevs_adj.append('SSD') 
        elif a == 'SUD':
             abbrevs_adj.append('SDN') 
        elif a == 'LIB':
             abbrevs_adj.append('LBY')
        elif a == 'ALG':
             abbrevs_adj.append('DZA') 
        elif a == 'MOR':
             abbrevs_adj.append('MAR') 
        elif a == 'SEY':
             abbrevs_adj.append('SYC') 
        elif a == 'AAB':
             abbrevs_adj.append('ATG') 
        elif a == 'ANG':
             abbrevs_adj.append('AGO')
        elif a == 'BAR':
             abbrevs_adj.append('BRB')
        elif a == 'BFO':
             abbrevs_adj.append('BFA')
        elif a == 'BHM':
             abbrevs_adj.append('BHS')
        elif a == 'BOS':
             abbrevs_adj.append('BIH')
        elif a == 'BOT':
             abbrevs_adj.append('BWA')
        elif a == 'BUI':
             abbrevs_adj.append('BDI')
        elif a == 'CAO':
             abbrevs_adj.append('CMR')
        elif a == 'CAP':
             abbrevs_adj.append('CPV')
        elif a == 'CEN':
             abbrevs_adj.append('CAF')
        elif a == 'CHA':
             abbrevs_adj.append('TCD')
        elif a == 'CON':
             abbrevs_adj.append('COG')
        elif a == 'CRO':
             abbrevs_adj.append('HRV')
        elif a == 'CZR':
             abbrevs_adj.append('CZE')
        elif a == 'DRC':
             abbrevs_adj.append('COD')
        elif a == 'EQG':
             abbrevs_adj.append('CNQ')
        elif a == 'GAM':
             abbrevs_adj.append('GMB')
        elif a == 'GDR':
             abbrevs_adj.append('DDR')
        elif a == 'GFR':
             abbrevs_adj.append('DEU')
        elif a == 'GRG':
             abbrevs_adj.append('GEO')
        elif a == 'GRN':
             abbrevs_adj.append('GRD')
        elif a == 'LAT':
             abbrevs_adj.append('LVA')
        elif a == 'LES':
             abbrevs_adj.append('LSO')
        elif a == 'LIT':
             abbrevs_adj.append('LTU')
        elif a == 'MAG':
             abbrevs_adj.append('MDG')
        elif a == 'MAS':
             abbrevs_adj.append('MUS')
        elif a == 'MAW':
             abbrevs_adj.append('MWI')
        elif a == 'MLD':
             abbrevs_adj.append('MDA')
        elif a == 'MNC':
             abbrevs_adj.append('MCO')
        elif a == 'MZM':
             abbrevs_adj.append('MOZ')
        elif a == 'NIG':
             abbrevs_adj.append('NGA')
        elif a == 'SAF':
             abbrevs_adj.append('ZAF')
        elif a == 'SIE':
             abbrevs_adj.append('SLE')
        elif a == 'SKN':
             abbrevs_adj.append('KNA')
        elif a == 'SLO':
             abbrevs_adj.append('SVK')
        elif a == 'SLU':
             abbrevs_adj.append('LCA')
        elif a == 'SNM':
             abbrevs_adj.append('SMR')
        elif a == 'SVG':
             abbrevs_adj.append('VCT')
        elif a == 'SWA':
             abbrevs_adj.append('SWZ')
        elif a == 'TAW':
             abbrevs_adj.append('TWN')
        elif a == 'TAZ':
             abbrevs_adj.append('TZA')
        elif a == 'TOG':
             abbrevs_adj.append('TGO')
        elif a == 'YPR':
             abbrevs_adj.append('YDY')
        elif a == 'ZAM':
             abbrevs_adj.append('ZMB')
        elif a == 'ZIM':
             abbrevs_adj.append('ZWE')
    
    return abbrevs_adj