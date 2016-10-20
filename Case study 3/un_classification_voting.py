# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 09:49:30 2016

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc

from sklearn.metrics.pairwise import cosine_similarity

from scipy.cluster.vq import kmeans2

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
#import kPCA dataset
kpca = pd.read_csv(myPath + "doc2vec\\communities_by_kpca\\communities_kpca.csv",
                   index_col = 0)
#import doc2vec results
doc2vec = pd.read_csv(myPath + "doc2vec\\" + "doc2vec_results_kpca_rn.csv", 
                      index_col = 0)

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
ip_kpca.to_csv(myPath + 'ip_kpca.csv')

###############################################################################
###############################################################################
#cluster idealpoints into two
c = kmeans2(ip_kpca.Idealpoint.values, 2)
ip_kpca['cluster'] = c[1]

###############################################################################
###############################################################################
#set train and test sets for all 150 feature
X_train, X_test, y_train, y_test = train_test_split(
    ip_kpca[ip_kpca.columns[35:-6]].values, 
    ip_kpca.cluster.values, test_size=0.2, random_state=0)

#set train and test sets for three pca coordinates
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(
    ip_kpca[['kpca_x', 'kpca_y', 'kpca_z']].values, 
    ip_kpca.cluster.values, test_size=0.2, random_state=0)

###############################################################################
#let's look for a random forest classifier for doc vectors
rfc_params = {'n_estimators': [70, 85, 100], 'criterion': ['gini', 'entropy']}
rfc_model = RandomForestClassifier()
rfc_clf = GridSearchCV(rfc_model, rfc_params, verbose = 2, 
                       scoring= 'accuracy')
rfc_clf.fit(X_train, y_train)
rfc_best_params = rfc_clf.best_params_

rfc_model_bp = RandomForestClassifier(**rfc_best_params)
fc_model_bp = rfc_model_bp.fit(X_train, y_train)

y_pred_rfc = rfc_model_bp.predict(X_test)
rfc_accuracy_model = accuracy_score(y_test, y_pred_rfc)

#find optimal number of features
rfecv = RFECV(estimator=rfc_model_bp, step=1, cv=StratifiedKFold(y_train, 2),
              scoring='accuracy')
rfecv.fit(X_train, y_train)
print("Optimal number of features : %d" % rfecv.n_features_)
# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()


##let's look for a random forest classifier for pca coordinates
rfc_params_pca = {'n_estimators': [300, 350, 400], 'criterion': ['gini', 'entropy']}
rfc_model_pca = RandomForestClassifier()
rfc_clf_pca = GridSearchCV(rfc_model_pca, rfc_params_pca, verbose = 2, 
                       scoring= 'accuracy')
rfc_clf_pca.fit(X_train_pca, y_train_pca)
rfc_best_params_pca = rfc_clf_pca.best_params_

rfc_model_bp_pca = RandomForestClassifier(**rfc_best_params_pca)
rfc_model_bp_pca = rfc_model_bp_pca.fit(X_train_pca, y_train_pca)

y_pred_rfc_pca = rfc_model_bp_pca.predict(X_test_pca)
rfc_accuracy_model_pca = accuracy_score(y_test_pca, y_pred_rfc_pca)

###############################################################################
#let's look for a support vector machine model
svm_params = {'C': [0.5, 1, 1.5], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
svm_model = SVC()
svm_clf = GridSearchCV(svm_model, svm_params, verbose = 2, scoring= 'accuracy')
svm_clf.fit(X_train, y_train)
svm_best_params = svm_clf.best_params_

svm_model_bp = SVC(**svm_best_params)
svm_model_bp = svm_model_bp.fit(X_train, y_train)

y_pred_svm = svm_model_bp.predict(X_test)
svm_accuracy = accuracy_score(y_test, y_pred_svm)

###############################################################################
#let's look for a neural network model 


###############################################################################
###############################################################################
#function that returns mean accuracy against n different random int vectors
def f_accuracy_random(y_test, n_vectors):
    accuracies = np.zeros(n_vectors)
    for n in range(0,n_vectors):
        accuracies[n] = accuracy_score(y_test, np.random.randint(2, size = len(y_test)))
    return np.mean(accuracies)


###############################################################################
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