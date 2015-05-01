# prepare quant data for 2013/2014   
# pw is based on Performance Agriculture method
# RND is not filtered
# use myfy


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import bisect

v4vars = ['CashSecToCurrentLiab_adj', 'debtservicecoverage_adj', 'debttoebitda_adj', 'debttotangibleNW_adj', 'yearsinbusiness_adj', 'totalsales']


# read in quantitative data
hgc = pd.read_excel(u"H:\\work\\usgc\\2015\\quant\\quant.xlsx", sheetname = u'quant')

# missing counts by FY/model version 
hgc.ix[hgc.FY == 2014, v4vars].count()
hgc.ix[hgc.FY == 2013, v4vars].count()
hgc.ix[(hgc.FY == 2014) & (hgc.modelversion.isin([4, 4.1])), v4vars].count()
hgc.ix[(hgc.FY == 2014) & (hgc.modelversion.isin([3, 3.1])), v4vars].count()
hgc.ix[(hgc.FY == 2013) & (hgc.modelversion.isin([4, 4.1])), v4vars].count()
hgc.ix[(hgc.FY == 2013) & (hgc.modelversion.isin([3, 3.1])), v4vars].count()


# read in ratablenodata to filter data
#hgcRND = pd.read_excel(u"H:\\work\\usgc\\2015\\quant\\quant.xlsx", sheetname = u'RND')
#hgc = pd.merge(hgc, hgcRND, left_on = 'intArchiveID', right_on = 'ArchiveID', how = "left")

vars = [x for x in list(hgc) if 'adj' in x]
hgc[vars].describe().to_string() # quant variable summary

# read in supplementary data for Cash_Marketable_Securities
hgc_sup = pd.read_excel(u"H:\\work\\usgc\\2015\\quant\\quant.xlsx", sheetname = u'supplement2')  #Cash_Marketable_Securities = Cash + Time Deposits + Other Marketable Securities
hgc = pd.merge(hgc, hgc_sup, how = u'inner', on = u'intArchiveID')

# calculate Cash & Marketable Securities
hgc[u'CashMktSec'] = np.where(((hgc[u'Cash'].isnull()) & (hgc[u'TimeDeposits'].isnull()) &(hgc[u'OtherMarketableSecurities'].isnull())), np.nan, np.where(hgc[u'Cash'].isnull(), 0, hgc[u'Cash']) + np.where(hgc[u'TimeDeposits'].isnull(), 0, hgc[u'TimeDeposits']) + np.where(hgc[u'OtherMarketableSecurities'].isnull(), 0, hgc[u'OtherMarketableSecurities']))                                               
print 'new CashMktSec has %s non-missing, and %s missing' %(hgc[u'CashMktSec'].notnull().sum(),  hgc[u'CashMktSec'].isnull().sum())
hgc[u'CashMktSec'].describe()

# calculate the adjusted Fiscal Year (from Feb to Next Jan)
def newfyf(inputx):
	input_month = inputx.apply(lambda x: x.month)
	input_year = inputx.apply(lambda x: x.year)
	output_year = np.where(input_month >=2, input_year, input_year - 1)
	return output_year

hgc['newfy'] = newfyf(hgc['Final Form Date'])

      # Final Form Date    FY  newfy
# 0          2012-11-19  2013   2012
# 1          2013-04-19  2013   2013
# 2          2013-10-30  2013   2013
# 3          2012-12-28  2013   2012
# 4          2014-09-08  2014   2014
# 5          2012-12-11  2013   2012
# 6          2013-08-05  2013   2013
# 7          2014-06-23  2014   2014
# 8          2013-06-13  2013   2013
# 9          2012-11-09  2013   2012
# 10         2013-11-07  2014   2013

data_for_2014 = hgc.query('newfy == 2013')   #from 2013-02-01 to 2014-01-31

#################################################### read in default data ######################################################################
# read in default data and merge with quantitative data, one duplicated data was removed
dft = pd.read_excel(u"H:\\work\\usgc\\2015\\quant\\defaults_after_Oct2012_wo_canada.xlsx", sheetname = 'df4py')

#1 sort the default data by UEN + def_date in ascending order
dft.sort(['UEN', 'def_date'], ascending = [True, True])

#2 check who are duplicates, the output will be  "36030947 2014-09-29"
dft[dft.duplicated(['UEN'])]

#3 de-dup the duplicated data
df_dedup = dft.drop_duplicates(['UEN'])      # dft.groupby('UEN', group_keys=False).apply(lambda x: x.ix[x.def_date.idxmax()])


############################################################       merge to get default flag       ##################################################
# merge default with rating data
hgc = pd.merge(hgc, df_dedup, how='left', left_on = u'EntityUEN', right_on = 'UEN')
 
# calculate Default Flag and Default Year (to compare with FY) 
hgc['df_flag'] = np.where(hgc['def_date'].notnull(), 1, 0)
hgc['df_yr'] = np.where(hgc['def_date'].apply(lambda x: x.month) <= 10, hgc['def_date'].apply(lambda x: x.year), hgc['def_date'].apply(lambda x: x.year) + 1)


###################################################################### calculate time interval, Performance Window######################################

hgc['fyBegin'] = pd.Series(map(lambda x: np.datetime64(str(x) + '-11-01'), hgc.newfy))
hgc['dura_fybegin_attst'] = (hgc[u'Final Form Date'] - hgc[u'fyBegin']) / np.timedelta64(1, 'M')
hgc['dura_fs_attst'] = (hgc[u'Final Form Date'] - hgc[u'Financial Statement Date']) / np.timedelta64(1, 'M')
hgc['dura_fybegin_attst_abs'] = abs(hgc['dura_fybegin_attst'])
hgc['dura_attst_df'] = (hgc[u'def_date'] - hgc[u'Final Form Date']) / np.timedelta64(1, 'M')
hgc['dura_fs_df'] = (hgc[u'def_date'] - hgc[u'Financial Statement Date']) / np.timedelta64(1, 'M')

hgc['pw'] = 0

hgc['pw'][(hgc['df_flag'] == 0) & ((-9 <= hgc['dura_fybegin_attst']) & (hgc['dura_fybegin_attst'] <0))] = 1    #ratings dated up to 9 months prior to the beginning of the fiscal year
hgc['pw'][(hgc['df_flag'] == 0) & ((0 <= hgc['dura_fybegin_attst']) & (hgc['dura_fybegin_attst'] <= 3))] = 2
hgc['pw'][(hgc['df_flag'] == 1) & (hgc.FY == hgc.df_yr) & ((3 <= hgc['dura_attst_df']) & (hgc['dura_attst_df'] <= 12))] = 1
hgc['pw'][(hgc['df_flag'] == 1) & (hgc.FY == hgc.df_yr) & ((12 < hgc['dura_attst_df']) & (hgc['dura_attst_df'] <= 18))] = 2
hgc['pw'][(hgc['df_flag'] == 1) & (hgc.FY == hgc.df_yr) & ((0 <= hgc['dura_attst_df']) & (hgc['dura_attst_df'] < 3))] = 3
hgc['pw'][(hgc['df_flag'] == 1) & (hgc.FY == hgc.df_yr) & ((18 < hgc['dura_attst_df']) | (hgc['dura_attst_df'].isnull()))] = 4
hgc['pw'][((hgc['df_flag'] == 0) & (hgc['dura_fs_attst'] > 15)) | ((hgc['df_flag'] == 1) & ((hgc['dura_attst_df'] <= 0) | (hgc['dura_fs_df'] > 24)))] = 9
hgc['mkey'] = hgc[u'EntityUEN'].map(str) + hgc[u'newfy'].map(str)

# save hgc + cash&mktSec + df_flag + pw related info as final data 
#hgc.to_excel(u"H:\\work\\usgc\\2015\\quant\\hgc_quant_output_dataV3.xlsx", sheet_name = "hgc quant out v3") 

##########################################################    Dedup data after PW    #################################################################

# I dont wannt change hgc data order, so sort it to another data named hgcsort
hgcsort = hgc.sort(['mkey', 'df_flag', 'pw', 'dura_fybegin_attst_abs'], ascending = [True, False, True, True])
hgc_after_pw = hgcsort.ix[((hgcsort['pw'] != 0) & (hgcsort['pw'] != 9)), :].drop_duplicates(['mkey'])
#hgc_after_pw.to_excel(u"H:\\work\\usgc\\2015\\quant\\hgc_after_pw_output_dataV3.xlsx", sheet_name = "hgc_after_pw_dataV3") 
 

##########################################################    Analysis after PW Dedup    #################################################################
hgc_after_pw.query('FY == 2014').df_flag.value_counts(dropna = False)  	# {0:3800, 1:34}
hgc_after_pw.query('FY == 2013').df_flag.value_counts(dropna = False)  	# {0:4810, 1:38}

# check missing counts after PW
hgc_after_pw.ix[hgc_after_pw.FY == 2014, v4vars].count()
hgc_after_pw.ix[hgc_after_pw.FY == 2013, v4vars].count()
hgc_after_pw.ix[(hgc_after_pw.FY == 2014) & (hgc_after_pw.modelversion.isin([4, 4.1])), v4vars].count()
hgc_after_pw.ix[(hgc_after_pw.FY == 2014) & (hgc_after_pw.modelversion.isin([3, 3.1])), v4vars].count()
hgc_after_pw.ix[(hgc_after_pw.FY == 2013) & (hgc_after_pw.modelversion.isin([4, 4.1])), v4vars].count()
hgc_after_pw.ix[(hgc_after_pw.FY == 2013) & (hgc_after_pw.modelversion.isin([3, 3.1])), v4vars].count()


















