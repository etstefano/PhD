# -*- coding: utf-8 -*-
"""
Created on Mon May 02 08:49:07 2016
This script compares dyadic data with country similarities

@author: S
"""
import sys  
reload(sys)  
sys.setdefaultencoding('utf8')

import os 
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mutual_info_score

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

dyads = pd.read_csv(myPath + "Dyadicdata.tab", sep = "\t")
meta_mat = pd.read_csv(myPath + "doc2vec2\\"  + "meta_mat.csv", index_col = 0)

#create edge list from meta matrix
edge_list = mat2eg(meta_mat)
edge_list['year'] = edge_list.country1.apply(lambda x: int(x[-4:]))
edge_list['iso_code1'] = edge_list.country1.apply(lambda x: x[:3])
edge_list['iso_code2'] = edge_list.country2.apply(lambda x: x[:3])
#filter and add isocodes to dyads
dyads = dyads[dyads.year.isin(edge_list.year.unique())]
c_codes = pd.read_csv(myPath + "COW country codes.csv")
stateAbb1 = [c_codes[c_codes.CCode == i].StateAbb.values[0] for i in dyads.ccode1.values.tolist()]
stateAbb2 = [c_codes[c_codes.CCode == i].StateAbb.values[0] for i in dyads.ccode2.values.tolist()]
dyads['stateAbb1'] = stateAbb1
dyads['stateAbb2'] = stateAbb2
del stateAbb1,stateAbb2
dyads['iso_code1'] = f_match_abbrevs(dyads.stateAbb1.values.tolist(), country_dict)
dyads['iso_code2'] = f_match_abbrevs(dyads.stateAbb2.values.tolist(), country_dict)

merge = pd.merge(dyads, edge_list, how='inner', on=['year', 'iso_code1', 'iso_code2'])
merge.to_csv(myPath + 'dyadic_merged.csv')

spearman_global_absidealdiff = spearmanr(merge.edge.values, merge.absidealdiff.values)
spearman_global_s2un = spearmanr(merge.edge.values, merge.s2un.values)
spearman_global_s3un = spearmanr(merge.edge.values, merge.s3un.values)

spear_ts_1 = f_spearman_ts(merge, value2 = 'absidealdiff')
spear_ts_2 = f_spearman_ts(merge, value2 = 's2un')
spear_ts_3 = f_spearman_ts(merge, value2 = 's3un')

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
line1, = ax1.plot(spear_ts_1.keys(), spear_ts_1.values())
line2, = ax1.plot(spear_ts_1.keys(), spear_ts_3.values())
ax1.set_ylim(-0.5, 0.5)
ax1.set_xlabel('Year')
ax1.set_ylabel('Spearman correlation coefficient')
ax1.legend((line1, line2), ('IdealPoint Abs Difference', 'Affinity Index'))
fig1.save()

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
line2, = ax1.plot(spear_ts_1.keys()[1:], spear_ts_3.values()[1:], c = '0')
line1, = ax1.plot(spear_ts_1.keys()[1:], np.zeros(len(spear_ts_1.keys()[1:])),
                  linestyle = "--", c = '0.7')
ax1.set_ylim(-0.5, 0.5)
ax1.set_xlabel('Year', fontsize = 14)
ax1.set_ylabel('Spearman correlation coefficient', fontsize = 14)

###############################################################################
###############################################################################
def f_spearman_ts(merge, value2 = 's2sun'):
    d = dict()
    for yy in merge.year.unique():
        merge_temp = merge[merge.year == yy]
        d[yy] = spearmanr(merge_temp.edge.values, merge_temp[value2].values)[0]
    
    return d


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
            if meta_mat.index[i][-4:] == meta_mat.columns[j][-4:]:
                country1.append(meta_mat.index[i])
                country2.append(meta_mat.columns[j])
                edge.append(meta_mat.ix[meta_mat.index[i]][meta_mat.columns[j]])
    
    d = {'country1' : country1,
         'country2' : country2,
         'edge': edge}
         
    return  pd.DataFrame(d)

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
        else:
             abbrevs_adj.append(a)
    
    return abbrevs_adj