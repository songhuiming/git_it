# prepare quant data for FY2014(2013/11/1): good FFD[2013Feb1, 2014Jan31], bad DFD[2013Nov1, 2014Oct31]  
# RND is not filtered
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import bisect

pdIntval = [0.0003, 0.0007, 0.0011, 0.0019, 0.0032, 0.0054, 0.0091, 0.0154, 0.0274, 0.0516, 0.097, 0.1823, 1]
msRating = ['I-2', 'I-3', 'I-4', 'I-5', 'I-6', 'I-7', 'S-1', 'S-2', 'S-3', 'S-4', 'P-1', 'P-2', 'P-3', 'D-1']
pcRR = [10, 15, 20, 25, 30, 35, 40, 45, 46, 47, 50, 51, 52]
ranking = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
newMapPD = [0.0003, 0.0005, 0.0009, 0.0015, 0.0026, 0.0043, 0.0073, 0.0123, 0.0214, 0.0395, 0.0743, 0.1397, 0.2412]
oldMapPD = [0.0004, 0.0007, 0.0013, 0.0025, 0.004, 0.0067, 0.0107, 0.018, 0.0293, 0.0467, 0.0933, 0.18, 0.3333] 

v4vars = ['cash', 'timedeposits', 'othermarketablesecurities', 'currentratio_adj', 'cashsectocurrentliab_adj', 'debtservicecoverage_adj', 'debttoebitda_adj', 'debttotangiblenw_adj', 'yearsinbusiness_adj', 'totalsales']
v4vars2 = ['currentassets', 'totalassets', 'currentliabilities', 'totalnoncurrentliabilities', 'debt', 'tangiblenetworth', 'totalsales', 'netprofit', 'ebitda', 'debtservicetotalamount']
v4vars3 = [u'entityuen',  u'financial_statement_date', u'final_form_date', u'us_sic', ]
#{'entityuen': 'uen', 'financial_statement_date': 'fin_stmt_dt', }

# read in quantitative data
#hgc = pd.read_excel(u"H:\\work\\usgc\\2015\\quant\\quant.xlsx", sheetname = u'quant')
#hgc.to_pickle(u"H:\\work\\usgc\\2015\\quant\\2014\\FACT2014")
hgc = pd.read_pickle(u"H:\\work\\usgc\\2015\\quant\\2014\\FACT2014")
hgc.columns = [x.replace(' ', '_').lower() for x in list(hgc)]
hgc_rename = {'currentassets': 'cur_ast_amt', 'totalassets': 'tot_ast_amt', 'currentliabilities': 'cur_liab_amt', 'debt': 'tot_debt_amt', 'tangiblenetworth': 'tangible_net_worth_amt', 'totalsales': 'tot_sales_amt', 'ebitda': 'ebitda_amt', 'years_in_business_c': 'yrs_in_bus', 'debtservicetotalamount': 'ds_amt'}
hgc['tot_liab_amt'] = hgc.currentliabilities + hgc.totalnoncurrentliabilities
hgc = hgc.rename(columns = hgc_rename)

# new: use (ebitda - substaxliabilitymemo)/debtserviceamount as dsc calculation 
hgc_dsc_sup = pd.read_excel("H:\\work\\usgc\\2015\\quant\\2015_supp\\HGC FACT data (Jul 22 2015).xlsx", sheetname = "GC US data")
hgc_dsc_sup.columns = [x.replace(' ', '_').lower() for x in list(hgc_dsc_sup)]
hgc_dsc_sup = hgc_dsc_sup.ix[:, ["intarchiveid", "substaxliabilitymemo"]]
hgc = pd.merge(hgc, hgc_dsc_sup, on = 'intarchiveid', how = 'left')
hgc["ebitda_substax"] = np.where(hgc.substaxliabilitymemo.notnull(), hgc.ebitda_amt - hgc.substaxliabilitymemo, hgc.ebitda_amt)

hgc['quant_ranking'] = hgc.quantitativerating.map(dict(zip(msRating, ranking)))
hgc['final_ranking'] = hgc.final_rating.map(dict(zip(msRating, ranking)))

#################################################### read in supplementary data ######################################################################
F2014FACT_sup_cols = ['archive_statement_id', 'archiveid', 'netincome', 'totaloperatingincome', 'cv_debt_ebitda_adj', 'cv_debttotangiblenw_adj']
#f2014fact_sup_data = pd.read_excel("H:\\work\\usgc\\2015\\quant\\2015_supp\\HGC_FACT_Financial_Variable_V2.xlsx")
#f2014fact_sup_data.columns = [x.replace(' ', '_').lower() for x in list(f2014fact_sup_data)]
#f2014fact_sup_data.to_pickle("H:\\work\\usgc\\2015\\quant\\2015_supp\\f2014fact_sup_dat")
f2014fact_sup_data = pd.read_pickle("H:\\work\\usgc\\2015\\quant\\2015_supp\\f2014fact_sup_dat")
f2014fact_sup_data_1 = f2014fact_sup_data.ix[:, F2014FACT_sup_cols]
F2014FACT_sup_cols_rename = {'archiveid': 'intarchiveid', 'netincome': 'net_inc_amt', 'totaloperatingincome': 'tot_optinc_amt', 'cv_debt_ebitda_adj': 'debt_to_ebitda_rto', 'cv_debttotangiblenw_adj': 'debt_to_tnw_rto'}
f2014fact_sup_data_1 = f2014fact_sup_data_1.rename(columns = F2014FACT_sup_cols_rename)

hgc = pd.merge(hgc, f2014fact_sup_data_1, on = 'intarchiveid', how = 'left')

#####  hgc to merge in financial statements and calculate ratios 
## table19_calc:  Non-Debt Based Ratios ( except Net Margin, EBITDA Margin, EBIT Margin)
def table19_calc(x, y):
	return np.where((x.isnull()) | (y.isnull()), np.nan, np.where((x == 0) & (y == 0), 0, np.where((x > 0) & (y == 0), 99999999, np.where((x < 0) & (y == 0), -99999999, np.where((x < 0) & (y < 0), -99999999, x / y )))))

## table20_calc:   Debt Based Ratios
def table20_calc(x, y):
	return np.where((x.isnull()) | (y.isnull()), np.nan, np.where((x == 0) & (y == 0), 0, np.where((x > 0) & (y == 0), np.nan, np.where((x > 0) & (y < 0), 99999999, x / y))))
	
## table21_calc:    Net Margin, EBITDA Margin, EBIT Margin
def table21_calc(x, y):
	return np.where((x.isnull()) | (y.isnull()), np.nan, np.where((x == 0) & (y == 0), 0, np.where((x > 0) & (y == 0), np.nan, np.where((x < 0) & (y < 0), -99999999, x / y))))

# years in business 	
def yib_calc(x, y):
	return (x - y) / np.timedelta64(1, 'Y')
	
# calculate some ratios
hgc['dsc'] = table19_calc(hgc.ebitda_substax, hgc.ds_amt)
hgc['dsc2'] = table19_calc(hgc.net_inc_amt, hgc.ds_amt)
hgc['dsc3'] = table19_calc(hgc.ebitda_amt, hgc.ds_amt)
hgc['cur_rto'] = table19_calc(hgc.cur_ast_amt, hgc.cur_liab_amt) 
hgc['net_margin_rto'] = table19_calc(hgc.net_inc_amt, hgc.tot_sales_amt)


#################################################### end of read in supplementary data ######################################################################

# missing counts by FY/model version 
# hgc.ix[hgc.fy == 2014, v4vars].count()
# hgc.ix[hgc.fy == 2013, v4vars].count()
# hgc.ix[(hgc.fy == 2014) & (hgc.modelversion.isin([4, 4.1])), v4vars].count()
# hgc.ix[(hgc.fy == 2014) & (hgc.modelversion.isin([3, 3.1])), v4vars].count()
# hgc.ix[(hgc.fy == 2013) & (hgc.modelversion.isin([4, 4.1])), v4vars].count()
# hgc.ix[(hgc.fy == 2013) & (hgc.modelversion.isin([3, 3.1])), v4vars].count()


# read in ratablenodata to filter data
hgcRND = pd.read_excel(u"H:\\work\\usgc\\2015\\quant\\quant.xlsx", sheetname = u'RND')
hgc = pd.merge(hgc, hgcRND, on = 'intarchiveid', how = "left")
hgc = hgc.ix[hgc.RND == 'No', :] 

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
df_dedup.columns = [u'entityuen', u'def_date']

############################################################       merge to get default flag       ##################################################
# merge default with rating data
hgc = pd.merge(hgc, df_dedup, how='left', on = 'entityuen')
 
# calculate Default Flag and Default Year (to compare with FY) 
hgc['default_flag'] = np.where(hgc['def_date'].notnull(), 1, 0)
hgc['df_yr'] = np.where(hgc['def_date'].apply(lambda x: x.month) <= 10, hgc['def_date'].apply(lambda x: x.year), hgc['def_date'].apply(lambda x: x.year) + 1)


################################################################ calculate time interval, Performance Window for 2014 ######################################
data_for_2014 = hgc.query('(newfy == 2013 & default_flag == 0) | (newfy == 2013 & default_flag == 1 & df_yr == 2014)')   #FFD[2013-02-01 2014-01-31]  DFD[2013-11-1 2014-10-31]
 
data_for_2014['fyBegin'] = map(lambda x: np.datetime64(str(x) + '-11-01'), data_for_2014.ix[:, 'newfy'])
data_for_2014['dura_fybegin_attst'] = (data_for_2014.ix[:, u'final_form_date'] - data_for_2014.ix[:, u'fyBegin']) / np.timedelta64(1, 'M')
data_for_2014['dura_fs_attst'] = (data_for_2014.ix[:, u'final_form_date'] - data_for_2014.ix[:, u'financial_statement_date']) / np.timedelta64(1, 'M')
data_for_2014['dura_fybegin_attst_abs'] = abs(data_for_2014.ix[:, 'dura_fybegin_attst'])
data_for_2014['dura_attst_df'] = (data_for_2014.ix[:, u'def_date'] - data_for_2014.ix[:, u'final_form_date']) / np.timedelta64(1, 'M')
data_for_2014['dura_fs_df'] = (data_for_2014.ix[:, u'def_date'] - data_for_2014.ix[:, u'financial_statement_date']) / np.timedelta64(1, 'M')

data_for_2014.ix[:, 'pw'] = 0

data_for_2014.ix[(data_for_2014['default_flag'] == 0) & ((-9 <= data_for_2014['dura_fybegin_attst']) & (data_for_2014['dura_fybegin_attst'] <0)), 'pw'] = 1     
data_for_2014.ix[(data_for_2014['default_flag'] == 0) & ((0 <= data_for_2014['dura_fybegin_attst']) & (data_for_2014['dura_fybegin_attst'] <= 3)), 'pw'] = 2
data_for_2014.ix[(data_for_2014['default_flag'] == 1) & ((3 <= data_for_2014['dura_attst_df']) & (data_for_2014['dura_attst_df'] <= 12)), 'pw'] = 1
data_for_2014.ix[(data_for_2014['default_flag'] == 1) & ((12 < data_for_2014['dura_attst_df']) & (data_for_2014['dura_attst_df'] <= 18)), 'pw'] = 2
data_for_2014.ix[(data_for_2014['default_flag'] == 1) & ((0 <= data_for_2014['dura_attst_df']) & (data_for_2014['dura_attst_df'] < 3)), 'pw'] = 3
data_for_2014.ix[(data_for_2014['default_flag'] == 1) & ((18 < data_for_2014['dura_attst_df']) | (data_for_2014['dura_attst_df'].isnull())), 'pw'] = 4
data_for_2014.ix[((data_for_2014['default_flag'] == 0) & (data_for_2014['dura_fs_attst'] > 15)) | ((data_for_2014['default_flag'] == 1) & ((data_for_2014['dura_attst_df'] <= 0) | (data_for_2014['dura_fs_df'] > 24))), 'pw'] = 9
data_for_2014['mkey'] = data_for_2014.ix[:, u'entityuen'].map(str) + data_for_2014.ix[:, u'newfy'].map(str)
	
##########################################################    Dedup data after PW    #################################################################

# I dont wannt change data_for_2014 data order, so sort it to another data named data_for_2014_sort
data_for_2014_sort = data_for_2014.sort(['mkey', 'default_flag', 'pw', 'dura_fybegin_attst_abs'], ascending = [True, False, True, True])
data_2014_after_pw = data_for_2014_sort.ix[((data_for_2014_sort['pw'] != 0) & (data_for_2014_sort['pw'] != 9)), :].drop_duplicates(['mkey'])     # df = {1: 49, 0: 4744} 

# read in M&I data, check how many of them are M&I
legacy_mi = pd.read_excel(u"H:\\work\\usgc\\2015\\quant\\2013\\2013_combined_data.xlsx", sheetname = u'legacy_mi_uen_new')
data_2014_after_pw['mi_flag'] = np.where(data_2014_after_pw.entityuen.isin(legacy_mi.uen), 1, 0)        #{1: 2775, 0: 2018}

print data_2014_after_pw.mi_flag.value_counts(dropna = False).to_string()
print data_2014_after_pw.default_flag.value_counts(dropna = False).to_string()

# US SIC
sic_indust = pd.read_excel("H:\\work\\usgc\\2015\\quant\\SIC_Code_List.xlsx", sheetname = "sic_indust") 
data_2014_after_pw['indust'] = data_2014_after_pw.us_sic.replace(dict(zip(sic_indust.sic_code, sic_indust.indust)))
data_2014_after_pw['sector_group'] = data_2014_after_pw.us_sic.replace(dict(zip(sic_indust.sic_code, sic_indust.sector_group))) 

data2014_pw_sic = data_2014_after_pw.query('sector_group not in ["NONP", "AGRI", "AGSM", "OTHR", "FOST"]')      # {0: 1169, 1: 14}

print pd.crosstab(data2014_pw_sic.mi_flag, data2014_pw_sic.default_flag)

# 'cur_ast_amt', 'tot_ast_amt', 'cur_liab_amt', 'tot_liab_amt', 'tot_debt_amt', 'net_worth_amt', 'tangible_net_worth_amt', 'tot_sales_amt', 'net_inc_amt', 'ebitda_amt', 'tot_debt_srvc_amt', 'ebit_amt', 'cash_and_secu_amt', 'debt_srvc_cov_rto', 'debt_to_tnw_rto', 'debt_to_ebitda_rto'

# data2014_pw_sic.ix[:, final_model_vars].count()
# Out[206]: 
# cur_ast_amt               3639
# tot_ast_amt               3639
# cur_liab_amt              3637
# tot_liab_amt              3166
# tot_debt_amt              3637
# net_worth_amt                0
# tangible_net_worth_amt    3635
# tot_sales_amt             3633
# net_inc_amt               3632
# ebitda_amt                3633
# dsc                       3434
# yrs_in_bus                3611
# debt_to_tnw_rto           3601
# debt_to_ebitda_rto        3602
# net_margin_rto            3632
# cur_rto                   3637

data2014_pw_sic['yeartype'] = np.where(data2014_pw_sic.mi_flag == 1, '2014MI', '2014HBC')
data2014_pw_sic = data2014_pw_sic.rename(columns = {'def_date': 'default_date', 'entityuen': 'uen', u'financial_statement_date': 'fin_stmt_dt'})

common_vars = ['sk_entity_id', 'uen', 'final_form_date', 'us_sic', 'default_flag', 'default_date', 'yeartype', 'sector_group', 'quant_ranking', 'final_ranking', u'fin_stmt_dt', 'intarchiveid', 'substaxliabilitymemo', 'ds_amt']
final_model_vars = ['cur_ast_amt', 'tot_ast_amt', 'cur_liab_amt', 'tot_liab_amt', 'tot_debt_amt', 'net_worth_amt', 'tangible_net_worth_amt', 'tot_sales_amt', 'net_inc_amt', 'ebitda_amt', 'dsc', 'yrs_in_bus', 'debt_to_tnw_rto', 'debt_to_ebitda_rto', 'net_margin_rto', 'cur_rto']

data2014_pw_sic.ix[:, common_vars + final_model_vars].to_excel("H:\\work\\usgc\\2015\\quant\\2015_supp\\F2014ALL_4_model.xlsx") 
 






