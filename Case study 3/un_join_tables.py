# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 08:35:51 2016

@author: S
"""

import sys  
reload(sys)  
sys.setdefaultencoding('utf8')

import re
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
country_dict['BHM'] = iso3166.Country(name = u'Bahamans', alpha2='BH', 
    alpha3='BHM', numeric='1004', apolitical_name=u'Bahamas')
country_dict['HAI'] = iso3166.Country(name = u'Haiti', alpha2='HA', 
    alpha3='HAI', numeric='1005', apolitical_name=u'Haiti')
country_dict['TRI'] = iso3166.Country(name = u'Trinidad and Tobago', alpha2='TT', 
    alpha3='TRI', numeric='1006', apolitical_name=u'Trinidad and Tobago')
###############################################################################
    
def hasNumbers(inputString):
    return bool(re.search(r'\d', inputString))

myPath = "C:\\Users\\S\\Documents\\Uni\\UNspeeches\\"
#import dataset
data = pd.read_csv(myPath + "doc2vec\\" + "doc2vec_results.csv", index_col = 0)

#select only countries
inds = []
for i in data.index:
    try:
        inds.append(i[0].isupper())
    except TypeError:
        inds.append(False)
        continue

data_c = data[inds]

#join with dataframe containing results from kPCA
data_kpca = pd.read_csv(myPath + "doc2vec\\communities_by_kpca\\" + "communities_kpca.csv", 
                   index_col = 0)

data_joined = data_c.join(data_kpca)
data_joined.to_csv(myPath + "doc2vec\\" + "doc2vec_results_kpca.csv")

#change column names
new_colnames = ['feature_' + str(i) for i in data_joined.columns[:150]]
new_colnames = new_colnames + data_joined.columns[150:].tolist()
data_joined.columns = new_colnames

data_joined.to_csv(myPath + "doc2vec\\" + "doc2vec_results_kpca_rn.csv")
###############################################################################
#export for each year
years_range = range(1962, 2015)

for yy in years_range:
    print yy
    index_y = []
    for ix in data.index: 
        try:
            if str(yy) in ix:
                index_y.append(ix)
        except TypeError:
            continue
            
    data_joined_y = data_joined.ix[index_y]
    data_joined_y.to_csv(myPath + "doc2vec\\" + str(yy) + "_doc2vec_results_kpca_rn.csv")
###############################################################################    


###############################################################################
###############################################################################
#integrate info on un voting   
data_joined = pd.read_csv(myPath + "doc2vec\\" + "doc2vec_results_kpca_rn.csv",
                          index_col = 0)
sessions = []
for ind in data_joined.index:
    if len(ind) == 12:
        sessions.append(float(ind[5:7]))
    else:
        sessions.append(float(ind[4:6]))
data_joined['session'] = sessions

un_voting = pd.read_csv(myPath + "RawVotingdata.tab", delimiter = "\t")
un_voting = un_voting[un_voting.session >= min(data_joined.session)]
un_voting = un_voting.dropna(subset = ["ccode"])
#match country code with iso alpha3
c_codes = pd.read_csv(myPath + "COW country codes.csv")
c_codes.columns = ['StateAbb', 'ccode', 'StateNme']
iso_codes = list()
for c in un_voting.index:
    iso_codes.append(c_codes[c_codes.ccode == un_voting.ix[c]['ccode']]['StateAbb'].values[0]) 
un_voting['iso_code'] = iso_codes
#filter only for session in the speech dataset
cond_ss = [ii in data_joined.session.unique() for ii in un_voting.session]
un_voting = un_voting[cond_ss]
#add year column
years = []
for ss in un_voting.session:
    try:
        years.append(data_joined[data_joined.session == ss].year.values[0])
    except IndexError:
        continue
un_voting['year'] = years
un_voting.to_csv(myPath + "un_voting\\" + "un_voting.csv")


un_newindex = [un_voting.ix[un_voting.index[c]]['iso_code'] + '_' + str(c) for c in range(0, len(un_voting.iso_code))]
un_voting['new_ind'] = un_newindex
###############################################################################
#create new dataframe for each year
for yy in un_voting.year.unique()[36:]:
    print yy
    un_voting_temp = un_voting[un_voting.year == yy]
    cols = ["rcid_" + str(int(rr)) for rr in un_voting_temp.rcid.unique()] + ["iso_code"]
    un_pivot = pd.DataFrame(index = range(0,len(un_voting_temp.iso_code.unique())), columns = cols)
    #for each iso code, for each  rcid in temp, fill values
    for i_code in range(0,len(un_voting_temp.iso_code.unique())):
        un_pivot.ix[i_code]['iso_code'] = un_voting_temp.iso_code.unique()[i_code]
        for rc in range(0,len(un_voting_temp.rcid.unique())):
            try:
                un_pivot.ix[i_code][un_pivot.columns[rc]] = un_voting_temp[(un_voting_temp.iso_code == un_voting_temp.iso_code.unique()[i_code])][un_voting_temp.rcid == un_voting_temp.rcid.unique()[rc]].vote.values[0]
            except IndexError:
                 un_pivot.ix[i_code][un_pivot.columns[rc]] = 10                       
    #find names            
    names = [c_codes[c_codes.StateAbb == c].StateNme.values[0] for c in un_pivot.iso_code]
    un_pivot['names'] = names
    #export
    un_pivot.to_csv(myPath + "un_voting\\" + str(yy) + "_un_voting_piv.csv")

nn = un_voting[un_voting.session == 16].pivot(index = 'new_ind', columns = 'iso_code',
                                                values = 'vote')

un_voting.to_csv(myPath + "doc2vec\\"  + "un_voting_data.csv")
g = un_voting.groupby('rcid')
