#!/usr/bin/python

# As Miroslav email titlted "F2013 GC US sample for M&I" on Apr29 6:36PM

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
 
#read in the 2013 year start data 
hgc2013 = pd.read_excel(u"H:\\work\\usgc\\2015\\quant\\2013\\HBC_20110101_20131031.xlsx", sheetname = "HGC")

#1	For each borrower rated in Q1F2013 (between Nov. 1 2012 and Jan 31, 2013), take the rating that is closest to the beginning of F2013 (i.e. Nov. 1, 2012)   len = 55, no dups
r2013q1 = hgc2013.ix[(hgc2013.EFF_DT <= pd.datetime(2013,1,31)) & (hgc2013.EFF_DT >= pd.datetime(2012,11,1)), :]    # nodup, so no dedup

#2  •	For borrowers rated in F2013 between Feb. 1 and Oct. 31:  							 len = 3192 , 581 dups
# o	If there are multiple ratings, take the earliest one (the one that is closest to the beginning of the FY
# o	If there is only a single rating during this period, exclude from F2013 (as this rating would be used in construction of F2014 PW)

r2013rest = hgc2013.ix[(hgc2013.EFF_DT <= pd.datetime(2013,10,31)) & (hgc2013.EFF_DT >= pd.datetime(2012,2,1)), :]

# add duplicates indicator 
dup_unique_entity_id = r2013rest.ix[r2013rest.duplicated('SK_ENTITY_ID'), 'SK_ENTITY_ID']  
r2013rest.ix[:, 'dup_ind'] = np.where(r2013rest['SK_ENTITY_ID'].isin(dup_unique_entity_id), 1, 0)
#
r2013rest.ix[r2013rest.dup_ind == 1, ['SK_ENTITY_ID', 'EFF_DT']].to_excel(u"H:\\work\\usgc\\2015\\quant\\2013\\r2013rest_dups.xlsx", sheet_name = "dups")
