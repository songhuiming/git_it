# prepare quant data for 2013/2014   
# pw is based on Performance Agriculture method
# RND is not filtered


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import bisect

v4vars = ['CashSecToCurrentLiab_adj', 'debtservicecoverage_adj', 'debttoebitda_adj', 'debttotangibleNW_adj', 'yearsinbusiness_adj', 'totalsales']

# read in quantitative data
hgc = pd.read_excel(u"H:\\work\\usgc\\2015\\quant\\quant.xlsx", sheetname = u'quant')
# read in ratablenodata to filter data
#hgcRND = pd.read_excel(u"H:\\work\\usgc\\2015\\quant\\quant.xlsx", sheetname = u'RND')
#hgc = pd.merge(hgc, hgcRND, left_on = 'intArchiveID', right_on = 'ArchiveID', how = "left")
vars = [x for x in list(hgc) if 'adj' in x]
hgc[vars].describe().to_string() # quant variable summary
hgc_sup = pd.read_excel(u"H:\\work\\usgc\\2015\\quant\\quant.xlsx", sheetname = u'supplement2')  #Cash_Marketable_Securities = Cash + Time Deposits + Other Marketable Securities
hgc = pd.merge(hgc, hgc_sup, how = u'inner', on = u'intArchiveID')
# calculate Cash & Marketable Securities
hgc[u'CashMktSec'] = np.where(((hgc[u'Cash'].isnull()) & (hgc[u'TimeDeposits'].isnull()) &(hgc[u'OtherMarketableSecurities'].isnull())), np.nan, np.where(hgc[u'Cash'].isnull(), 0, hgc[u'Cash']) + np.where(hgc[u'TimeDeposits'].isnull(), 0, hgc[u'TimeDeposits']) + np.where(hgc[u'OtherMarketableSecurities'].isnull(), 0, hgc[u'OtherMarketableSecurities']))                                               
#hgc.ix[:, [u'EntityUEN', u'intArchiveID', u'CashMktSec']].to_excel(u"H:\\work\\usgc\\2015\\quant\\quant_results.xlsx", sheet_name = "new cash&mktSec")

# there are 150 ratings having 'Final Form Date' <= 'Financial Statement Date'
(hgc[u'Final Form Date'] > hgc[u'Financial Statement Date']).sum()        # 14338
(hgc[u'Final Form Date'] <= hgc[u'Financial Statement Date']).sum()       # 150 

#################################################### read in default data ######################################################################
# read in default data and merge with quantitative data, one duplicated data was removed
dft = pd.read_excel(u"H:\\work\\usgc\\2015\\quant\\defaults_after_Oct2012_wo_canada.xlsx", sheetname = 'df4py')
#1 sort the default data by UEN + def_date in ascending order
dft.sort(['UEN', 'def_date'], ascending = [True, True])
#2 check who are duplicates, the output will be  "36030947 2014-09-29"
dft[dft.duplicated(['UEN'])]
#3 de-dup the duplicated data
df_dedup = dft.drop_duplicates(['UEN'])      # dft.groupby('UEN', group_keys=False).apply(lambda x: x.ix[x.def_date.idxmax()])
######################################################################################################################################################

# merge default with rating data
hgc = pd.merge(hgc, df_dedup, how='left', left_on = u'EntityUEN', right_on = 'UEN')
hgc['def_date'].count()  #279 matched, this is verified in SAS
(hgc[u'Financial Statement Date'] - hgc['def_date']) / np.timedelta64(1, 'M')
hgc['df_flag'] = np.where(hgc['def_date'].notnull(), 1, 0)

######################################################################################################################################################

#check defaults on each year by def_date
hgc['def_date'].ix[hgc['def_date'] > 0].apply(lambda x: x.year).value_counts(dropna = False).sort_index()  # 2013:84, 2014:173
# FY and default year cross table
pd.crosstab(hgc.FY.ix[hgc['def_date'] > 0], hgc['def_date'].ix[hgc['def_date'] > 0].apply(lambda x: x.year))  

# check total obligors and defaults on each FY
hgc.ix[(hgc.FY==2013)].df_flag.value_counts(dropna = False)       								# 0: 6805  1: 177
hgc.ix[(hgc.FY==2013)].drop_duplicates([u'EntityUEN']).df_flag.value_counts(dropna = False)   	# 0: 4942  1: 115
hgc.ix[(hgc.FY==2014)].df_flag.value_counts(dropna = False)       								# 0: 5660  1: 92 
hgc.ix[(hgc.FY==2014)].drop_duplicates([u'EntityUEN']).df_flag.value_counts(dropna = False)   	# 0: 3883  1: 52

#test for 2014
hgc2014 = hgc.query('FY == 2014')
hgc2014sort = hgc2014.sort([u'EntityUEN', 'df_flag'], ascending = [True, False]).ix[:, [u'EntityUEN', 'df_flag']]


# datename = [x for x in list(hgc) if 'date' in x.lower()]
# hgc['fin_date'] = map(pd.Timestamp.date, hgc[u'Financial Statement Date'])
# hgc['final_form_date'] = map(pd.Timestamp.date, hgc[u'Final Form Date'])
# hgc['final_form_update_date'] = map(pd.Timestamp.date, hgc[u'Final Form Update Date'])

# pd.datetime.date(hgc['Final Form Date'].ix[(hgc.FY==2014) & (hgc.modelversion.isin([4, 4.1]))].min())
#check the defaults for each year
hgc.query('FY==2014 & modelversion in [3, 3.1]').df_flag.value_counts(dropna = False)  		#38
hgc.query('FY==2014 & modelversion in [4, 4.1]').df_flag.value_counts(dropna = False)  	#54
hgc['df_flag'].ix[(hgc.FY==2013)].value_counts(dropna = False)   #199


###################################################################### calculate time interval, Performance Window######################################
#          dura_fs_attst = INTCK('MONTH',FIN_STMT_DT,ATTST_DT);
#          dura_fybegin_attst = INTCK('MONTH',FYbegin,ATTST_DT);
#          dura_fybegin_attst_abs = abs(dura_fybegin_attst); * this is for use below (see algorithm, performing criteria);
#          dura_attst_df = INTCK('MONTH',ATTST_DT,DEFAULT_DT);
#          dura_fs_df = INTCK('MONTH',FIN_STMT_DT,DEFAULT_DT);
# fyBegin = pd.Series(map(lambda x: np.datetime64(str(x - 1) + '-11-01'), hgc['FY']))
# fyEnd = pd.Series(map(lambda x: np.datetime64(str(x) + '-10-31'), hgc['FY']))
# dura_fs_attst = (hgc[u'Final Form Date'] - hgc[u'Financial Statement Date']) / np.timedelta64(1, 'M')
# dura_fybegin_attst = (hgc[u'Final Form Date'] - fyBegin) / np.timedelta64(1, 'M')
# dura_fybegin_attst_abs = abs(dura_fybegin_attst)
# dura_attst_df = (hgc[u'def_date'] - hgc[u'Final Form Date']) / np.timedelta64(1, 'M')
# dura_fs_df = (hgc[u'def_date'] - hgc[u'Financial Statement Date']) / np.timedelta64(1, 'M')


#hgc['fyBegin'] = pd.Series(map(lambda x: np.datetime64(str(x - 1) + '-11-01'), hgc['FY']))

hgc['fyBegin'] = pd.Series(map(lambda x: np.datetime64(str(x) + '-11-01'), hgc['FY']))

hgc['dura_fybegin_attst'] = (hgc[u'Final Form Date'] - hgc[u'fyBegin']) / np.timedelta64(1, 'M')

hgc['dura_fs_attst'] = (hgc[u'Final Form Date'] - hgc[u'Financial Statement Date']) / np.timedelta64(1, 'M')

hgc['dura_fybegin_attst_abs'] = abs(hgc['dura_fybegin_attst'])
hgc['dura_attst_df'] = (hgc[u'def_date'] - hgc[u'Final Form Date']) / np.timedelta64(1, 'M')
hgc['dura_fs_df'] = (hgc[u'def_date'] - hgc[u'Financial Statement Date']) / np.timedelta64(1, 'M')

hgc['pw'] = 0
#hgc['pw'][(hgc['df_flag'] == 0) & ((-9 <= hgc['dura_fybegin_attst']) & (hgc['dura_fybegin_attst'] <= -1))] = 1
#hgc['pw'][(hgc['df_flag'] == 0) & ((0 <= hgc['dura_fybegin_attst']) & (hgc['dura_fybegin_attst'] <= 3))] = 2
hgc['pw'][(hgc['df_flag'] == 0) & ((-9 <= hgc['dura_fybegin_attst']) & (hgc['dura_fybegin_attst'] <0))] = 1    #ratings dated up to 9 months prior to the beginning of the fiscal year
hgc['pw'][(hgc['df_flag'] == 0) & ((0 <= hgc['dura_fybegin_attst']) & (hgc['dura_fybegin_attst'] <= 3))] = 2
hgc['pw'][(hgc['df_flag'] == 1) & ((3 <= hgc['dura_attst_df']) & (hgc['dura_attst_df'] <= 12))] = 1
hgc['pw'][(hgc['df_flag'] == 1) & ((13 <= hgc['dura_attst_df']) & (hgc['dura_attst_df'] <= 18))] = 2
hgc['pw'][(hgc['df_flag'] == 1) & ((0 <= hgc['dura_attst_df']) & (hgc['dura_attst_df']<= 3))] = 3
hgc['pw'][(hgc['df_flag'] == 1) & ((18 < hgc['dura_attst_df']) | (hgc['dura_attst_df'].isnull()))] = 4
hgc['pw'][((hgc['df_flag'] == 0) & (hgc['dura_fs_attst'] > 15)) | ((hgc['df_flag'] == 1) & ((hgc['dura_attst_df'] <= 0) | (hgc['dura_fs_df'] > 24)))] = 9
hgc['mkey'] = hgc[u'EntityUEN'].map(str) + hgc[u'FY'].map(str)

# save hgc + cash&mktSec + df_flag + pw related info as final data 
hgc.to_excel(u"H:\\work\\usgc\\2015\\quant\\quant_output_dataV1.xlsx", sheet_name = "hgc prepared data v1") 
######################################################################################################################################################


# I dont wannt change hgc data order, so sort it to another data named hgcsort
hgcsort = hgc.sort(['mkey', 'df_flag', 'pw', 'dura_fybegin_attst_abs'], ascending = [True, False, True, True])
hgc_after_pw = hgcsort.ix[((hgcsort['pw'] != 0) & (hgcsort['pw'] != 9)), :].drop_duplicates(['mkey'])
hgc_after_pw.isnull().sum()
hgc_after_pw.ix[:, vars].describe()



















