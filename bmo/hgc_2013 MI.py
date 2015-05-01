#!/usr/bin/python

# As Miroslav email titlted "F2013 GC US sample for M&I" on Apr29 6:36PM

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
writer = pd.ExcelWriter(u"H:\\work\\usgc\\2015\\quant\\2013\\MI_2013_Prepared_data.xlsx") 
#read in the 2013 year start data 
hgc = pd.read_excel(u"H:\\work\\usgc\\2015\\quant\\quant.xlsx", sheetname = u'quant')

#1	For each borrower rated in Q1F2013 (between Nov. 1 2012 and Jan 31, 2013), take the rating that is closest to the beginning of F2013 (i.e. Nov. 1, 2012)    
r2013q1 = hgc.ix[(hgc['Final Form Date'] <= pd.datetime(2013,1,31)) & (hgc['Final Form Date'] >= pd.datetime(2012,11,1)), :]        #1356    
r2013q1_sort = r2013q1.sort(['EntityUEN', 'Final Form Date'], ascending = [True, True])					#1356
r2013q1_final = r2013q1_sort.drop_duplicates('EntityUEN')                                              	#1285


#2  •	For borrowers rated in F2013 between Feb. 1 and Oct. 31:  							  
# o	If there are multiple ratings, take the earliest one (the one that is closest to the beginning of the FY
# o	If there is only a single rating during this period, exclude from F2013 (as this rating would be used in construction of F2014 PW)

r2013rest = hgc.ix[(hgc['Final Form Date'] <= pd.datetime(2013,10,31)) & (hgc['Final Form Date'] >= pd.datetime(2013,2,1)), :]				#5626

# add duplicates indicator 
dup_unique_uen = r2013rest.ix[r2013rest.duplicated('EntityUEN'), 'EntityUEN'].unique()  
r2013rest.ix[:, 'dup_ind'] = np.where(r2013rest['EntityUEN'].isin(dup_unique_uen), 1, 0)
#
r2013rest.ix[r2013rest.dup_ind == 1, ['EntityUEN', 'Final Form Date']].to_excel(writer, sheet_name = "2013Q2_Q4 >= 1 RR")
r2013rest_final = r2013rest.ix[r2013rest.dup_ind == 1, :].drop_duplicates('EntityUEN')														#986

mi2013 = pd.concat([r2013q1_final, r2013rest_final], axis = 0, ignore_index = True)          	# there are 196 dups, that is, rated in both q1 and rest
mi2013_dedup = mi2013.sort(['EntityUEN', 'Final Form Date']).drop_duplicates('EntityUEN')		#2075

#3  •	Check F2014 default list, to make sure that none of the borrowers that were flagged as non-defaults in this sample defaulted within one year of the Final Form date. 
#################################################### read in default data ######################################################################
# read in default data and merge with quantitative data, one duplicated data was removed
dft = pd.read_excel(u"H:\\work\\usgc\\2015\\quant\\defaults_after_Oct2012_wo_canada.xlsx", sheetname = 'df4py')
dft.columns = [x.lower() for x in list(dft)]

#1 sort the default data by UEN + def_date in ascending order
dft.sort(['uen', 'def_date'], ascending = [True, True])

#2 check who are duplicates, the output will be  "36030947 2014-09-29"
dft[dft.duplicated(['uen'])]

#3 de-dup the duplicated data
df_dedup = dft.drop_duplicates(['uen'])      # dft.groupby('uen', group_keys=False).apply(lambda x: x.ix[x.def_date.idxmax()])
 
#################################################################################################################################################
dft2014 = df_dedup.ix[(df_dedup.def_date <= pd.datetime(2014, 10, 31)) & (df_dedup.def_date >= pd.datetime(2013, 11, 1)), :]

mi2013_dedup.EntityUEN.isin(dft2014.uen).sum()       #33, meaning 33 from mi2013_dedup are in 2014 default list, remove them
mi2013_final = mi2013_dedup.ix[~mi2013_dedup['EntityUEN'].isin(dft2014.uen), :]             #2042 left

# 4   Defaulters:
# •	Start with the list of all borrowers that have defaulted between Nov. 1, 2012 and Oct. 31, 2013. This is the total defaulted population
# •	Take the earliest available rating in F2013 data; that will be the pre-default rating associated with defaulted borrower in F2013
dft2013 = dft.ix[(dft.def_date <= pd.datetime(2013, 10, 31)) & (dft.def_date >= pd.datetime(2012, 11, 1)), :]     
mi2013_final.ix[:, 'df_flag'] = np.where(mi2013_final.EntityUEN.isin(dft2013.uen), 1, 0)


######################################################################################################################################################
mi2013_final.df_flag.value_counts()           					# {0: 2018, 1: 24}
mi2013_final.to_excel(writer, sheet_name = "M&I 2013 Final")
writer.save()

 
 
 
 
 
 
 

 
