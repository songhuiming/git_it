# prepare quant data for FY2014(2013/11/1): good FFD[2013Feb1, 2014Jan31], bad DFD[2013Nov1, 2014Oct31]  
# RND is not filtered
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import bisect

v4vars = ['cashSecToCurrentLiab_adj', 'debtservicecoverage_adj', 'debttoebitda_adj', 'debttotangibleNW_adj', 'yearsinbusiness_adj', 'totalsales']

# read in quantitative data
hgc = pd.read_excel(u"H:\\work\\usgc\\2015\\quant\\quant.xlsx", sheetname = u'quant')
hgc.columns = [x.replace(' ', '_').lower() for x in list(hgc)]

# missing counts by FY/model version 
hgc.ix[hgc.fy == 2014, v4vars].count()
hgc.ix[hgc.fy == 2013, v4vars].count()
hgc.ix[(hgc.fy == 2014) & (hgc.modelversion.isin([4, 4.1])), v4vars].count()
hgc.ix[(hgc.fy == 2014) & (hgc.modelversion.isin([3, 3.1])), v4vars].count()
hgc.ix[(hgc.fy == 2013) & (hgc.modelversion.isin([4, 4.1])), v4vars].count()
hgc.ix[(hgc.fy == 2013) & (hgc.modelversion.isin([3, 3.1])), v4vars].count()


# read in M&I data
midf = pd.read_excel(u"H:\\work\\usgc\\2015\\quant\\2014\\Map_Legacy_MI.xlsx", sheetname = u'Map_Legacy_MI')
midf.columns = [x.replace(' ', '_').lower() for x in list(midf)]

# read in ratablenodata to filter data
#hgcRND = pd.read_excel(u"H:\\work\\usgc\\2015\\quant\\quant.xlsx", sheetname = u'RND')
#hgc = pd.merge(hgc, hgcRND, left_on = 'intArchiveID', right_on = 'ArchiveID', how = "left")
 

# read in supplementary data for cash_Marketable_Securities
hgc_sup = pd.read_excel(u"H:\\work\\usgc\\2015\\quant\\quant.xlsx", sheetname = u'supplement2')  #cash_Marketable_Securities = cash + Time Deposits + Other Marketable Securities
hgc_sup.columns = [x.replace(' ', '_').lower() for x in list(hgc_sup)]

hgc = pd.merge(hgc, hgc_sup, how = u'inner', on = u'intarchiveid')

# calculate cash & Marketable Securities
hgc[u'cashmktsec'] = np.where(((hgc[u'cash'].isnull()) & (hgc[u'timedeposits'].isnull()) &(hgc[u'othermarketablesecurities'].isnull())), np.nan, np.where(hgc[u'cash'].isnull(), 0, hgc[u'cash']) + np.where(hgc[u'timedeposits'].isnull(), 0, hgc[u'timedeposits']) + np.where(hgc[u'othermarketablesecurities'].isnull(), 0, hgc[u'othermarketablesecurities']))                                               
print 'new cashmktsec has %s non-missing, and %s missing' %(hgc[u'cashmktsec'].notnull().sum(),  hgc[u'cashmktsec'].isnull().sum())
hgc[u'cashmktsec'].describe()

# calculate the adjusted Fiscal Year (from Feb to Next Jan)
def newfyf(inputx):
	input_month = inputx.apply(lambda x: x.month)
	input_year = inputx.apply(lambda x: x.year)
	output_year = np.where(input_month >=2, input_year, input_year - 1)
	return output_year

hgc['newfy'] = newfyf(hgc['final_form_date'])

      # final_form_date    FY  newfy
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

#################################################### read in default data ######################################################################
# read in default data and merge with quantitative data, one duplicated data was removed
dft = pd.read_excel(u"H:\\work\\usgc\\2015\\quant\\defaults_after_Oct2012_wo_canada.xlsx", sheetname = 'df4py')
dft.columns = [x.lower() for x in list(dft)]

#1 sort the default data by uen + def_date in ascending order
dft.sort(['uen', 'def_date'], ascending = [True, True])

#2 check who are duplicates, the output will be  "36030947 2014-09-29"
dft[dft.duplicated(['uen'])]

#3 de-dup the duplicated data
df_dedup = dft.drop_duplicates(['uen'])      # dft.groupby('uen', group_keys=False).apply(lambda x: x.ix[x.def_date.idxmax()])


############################################################       merge to get default flag       ##################################################
# merge default with rating data
hgc = pd.merge(hgc, df_dedup, how='left', left_on = u'entityuen', right_on = 'uen')
 
# calculate Default Flag and Default Year (to compare with FY) 
hgc['df_flag'] = np.where(hgc['def_date'].notnull(), 1, 0)
hgc['df_yr'] = np.where(hgc['def_date'].apply(lambda x: x.month) <= 10, hgc['def_date'].apply(lambda x: x.year), hgc['def_date'].apply(lambda x: x.year) + 1)


################################################################ calculate time interval, Performance Window for 2014 ######################################
data_for_2014 = hgc.query('(newfy == 2013 & df_flag == 0) | (newfy == 2013 & df_flag == 1 & df_yr == 2014)')   #FFD[2013-02-01 2014-01-31]  DFD[2013-11-1 2014-10-31]
 
data_for_2014['fyBegin'] = map(lambda x: np.datetime64(str(x) + '-11-01'), data_for_2014.ix[:, 'newfy'])
data_for_2014['dura_fybegin_attst'] = (data_for_2014.ix[:, u'final_form_date'] - data_for_2014.ix[:, u'fyBegin']) / np.timedelta64(1, 'M')
data_for_2014['dura_fs_attst'] = (data_for_2014.ix[:, u'final_form_date'] - data_for_2014.ix[:, u'financial_statement_date']) / np.timedelta64(1, 'M')
data_for_2014['dura_fybegin_attst_abs'] = abs(data_for_2014.ix[:, 'dura_fybegin_attst'])
data_for_2014['dura_attst_df'] = (data_for_2014.ix[:, u'def_date'] - data_for_2014.ix[:, u'final_form_date']) / np.timedelta64(1, 'M')
data_for_2014['dura_fs_df'] = (data_for_2014.ix[:, u'def_date'] - data_for_2014.ix[:, u'financial_statement_date']) / np.timedelta64(1, 'M')

data_for_2014.ix[:, 'pw'] = 0

data_for_2014.ix[(data_for_2014['df_flag'] == 0) & ((-9 <= data_for_2014['dura_fybegin_attst']) & (data_for_2014['dura_fybegin_attst'] <0)), 'pw'] = 1     
data_for_2014.ix[(data_for_2014['df_flag'] == 0) & ((0 <= data_for_2014['dura_fybegin_attst']) & (data_for_2014['dura_fybegin_attst'] <= 3)), 'pw'] = 2
data_for_2014.ix[(data_for_2014['df_flag'] == 1) & ((3 <= data_for_2014['dura_attst_df']) & (data_for_2014['dura_attst_df'] <= 12)), 'pw'] = 1
data_for_2014.ix[(data_for_2014['df_flag'] == 1) & ((12 < data_for_2014['dura_attst_df']) & (data_for_2014['dura_attst_df'] <= 18)), 'pw'] = 2
data_for_2014.ix[(data_for_2014['df_flag'] == 1) & ((0 <= data_for_2014['dura_attst_df']) & (data_for_2014['dura_attst_df'] < 3)), 'pw'] = 3
data_for_2014.ix[(data_for_2014['df_flag'] == 1) & ((18 < data_for_2014['dura_attst_df']) | (data_for_2014['dura_attst_df'].isnull())), 'pw'] = 4
data_for_2014.ix[((data_for_2014['df_flag'] == 0) & (data_for_2014['dura_fs_attst'] > 15)) | ((data_for_2014['df_flag'] == 1) & ((data_for_2014['dura_attst_df'] <= 0) | (data_for_2014['dura_fs_df'] > 24))), 'pw'] = 9
data_for_2014['mkey'] = data_for_2014.ix[:, u'entityuen'].map(str) + data_for_2014.ix[:, u'newfy'].map(str)
	
##########################################################    Dedup data after PW    #################################################################

# I dont wannt change data_for_2014 data order, so sort it to another data named data_for_2014_sort
data_for_2014_sort = data_for_2014.sort(['mkey', 'df_flag', 'pw', 'dura_fybegin_attst_abs'], ascending = [True, False, True, True])
data_2014_after_pw = data_for_2014_sort.ix[((data_for_2014_sort['pw'] != 0) & (data_for_2014_sort['pw'] != 9)), :].drop_duplicates(['mkey'])     # df = {1: 49, 0: 4744} 

# check how many of them are M&I
data_2014_after_pw['mi_ind'] = np.where(data_2014_after_pw.entityuen.isin(midf.uen), 1, 0)        #{1: 2775, 0: 2018}

print data_2014_after_pw.mi_ind.value_counts(dropna = False).to_string()
print data_2014_after_pw.df_flag.value_counts(dropna = False).to_string()







#############  stop here  #################





##########################################################    Analysis after PW Dedup    #################################################################
# data_2014_after_pw.query('fy == 2014').df_flag.value_counts(dropna = False)  	 
# data_2014_after_pw.query('fy == 2013').df_flag.value_counts(dropna = False)  	 

# check missing counts after PW
# data_2014_after_pw.ix[data_2014_after_pw.fy == 2014, v4vars].count()
# data_2014_after_pw.ix[data_2014_after_pw.fy == 2013, v4vars].count()
# data_2014_after_pw.ix[(data_2014_after_pw.fy == 2014) & (data_2014_after_pw.modelversion.isin([4, 4.1])), v4vars].count()
# data_2014_after_pw.ix[(data_2014_after_pw.fy == 2014) & (data_2014_after_pw.modelversion.isin([3, 3.1])), v4vars].count()
# data_2014_after_pw.ix[(data_2014_after_pw.fy == 2013) & (data_2014_after_pw.modelversion.isin([4, 4.1])), v4vars].count()
# data_2014_after_pw.ix[(data_2014_after_pw.fy == 2013) & (data_2014_after_pw.modelversion.isin([3, 3.1])), v4vars].count()

















