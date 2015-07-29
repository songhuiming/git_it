#!/usr/bin/python

#1: read in FACT data
#2: read in the supplementary data by Zack, HGC_FACT_Financial_Variable_V2.xlsx
#3: default data
#4: Rules for M&I: As Miroslav email titlted "F2013 GC US sample for M&I" on Apr29 6:36PM

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
pdIntval = [0.0003, 0.0007, 0.0011, 0.0019, 0.0032, 0.0054, 0.0091, 0.0154, 0.0274, 0.0516, 0.097, 0.1823, 1]
msRating = ['I-2', 'I-3', 'I-4', 'I-5', 'I-6', 'I-7', 'S-1', 'S-2', 'S-3', 'S-4', 'P-1', 'P-2', 'P-3', 'D-1']
pcRR = [10, 15, 20, 25, 30, 35, 40, 45, 46, 47, 50, 51, 52]
ranking = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
newMapPD = [0.0003, 0.0005, 0.0009, 0.0015, 0.0026, 0.0043, 0.0073, 0.0123, 0.0214, 0.0395, 0.0743, 0.1397, 0.2412]
oldMapPD = [0.0004, 0.0007, 0.0013, 0.0025, 0.004, 0.0067, 0.0107, 0.018, 0.0293, 0.0467, 0.0933, 0.18, 0.3333] 
 
writer = pd.ExcelWriter(u"H:\\work\\usgc\\2015\\quant\\2013\\MI_2013_Prepared_data.xlsx") 
#read in the 2013 year start data 
#hgc = pd.read_excel(u"H:\\work\\usgc\\2015\\quant\\quant.xlsx", sheetname = u'quant')
#hgc.to_pickle(u"H:\\work\\usgc\\2015\\quant\\quant")
hgc = pd.read_pickle(u"H:\\work\\usgc\\2015\\quant\\quant")
hgc.columns = [x.replace(' ', '_').lower() for x in list(hgc)]

hgc['quant_ranking'] = hgc.quantitativerating.map(dict(zip(msRating, ranking)))
hgc['final_ranking'] = hgc.final_rating.map(dict(zip(msRating, ranking)))

hgc_rename = {'currentassets': 'cur_ast_amt', 'totalassets': 'tot_ast_amt', 'currentliabilities': 'cur_liab_amt', 'debt': 'tot_debt_amt', 'tangiblenetworth': 'tangible_net_worth_amt', 'totalsales': 'tot_sales_amt', 'ebitda': 'ebitda_amt', 'years_in_business_c': 'yrs_in_bus', 'debtservicetotalamount': 'ds_amt'}
hgc['tot_liab_amt'] = hgc.currentliabilities + hgc.totalnoncurrentliabilities
hgc = hgc.rename(columns = hgc_rename)


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
hgc['dsc'] = table19_calc(hgc.tot_optinc_amt, hgc.ds_amt)
hgc['cur_rto'] = table19_calc(hgc.cur_ast_amt, hgc.cur_liab_amt) 
hgc['net_margin_rto'] = table19_calc(hgc.net_inc_amt, hgc.tot_sales_amt)


#################################################### end of read in supplementary data ######################################################################


# read in Legacy M&I and only keep the data in M&I uen pool
legacy_mi = pd.read_excel(u"H:\\work\\usgc\\2015\\quant\\2013\\2013_combined_data.xlsx", sheetname = u'legacy_mi_uen_new')
legacy_mi.columns = [x.replace(' ', '_').lower() for x in list(legacy_mi)]
hgc = hgc.ix[hgc.entityuen.isin(legacy_mi.uen), :]

#1	For each borrower rated in Q1F2013 (between Nov. 1 2012 and Jan 31, 2013), take the rating that is closest to the beginning of F2013 (i.e. Nov. 1, 2012)    
r2013q1 = hgc.ix[(hgc['final_form_date'] <= pd.datetime(2013,1,31,23,59,59)) & (hgc['final_form_date'] >= pd.datetime(2012,11,1)), :]        #1356    
r2013q1_sort = r2013q1.sort(['entityuen', 'final_form_date'], ascending = [True, True])					#1356
r2013q1_final = r2013q1_sort.drop_duplicates('entityuen')                                              	#1285


#2  •	For borrowers rated in F2013 between Feb. 1 and Oct. 31:  							  
# o	If there are multiple ratings, take the earliest one (the one that is closest to the beginning of the FY
# o	If there is only a single rating during this period, exclude from F2013 (as this rating would be used in construction of F2014 PW)

r2013rest = hgc.ix[(hgc['final_form_date'] <= pd.datetime(2013,10,31,23,59,59)) & (hgc['final_form_date'] >= pd.datetime(2013,2,1)), :]				#5626

# add duplicates indicator 
dup_unique_uen = r2013rest.ix[r2013rest.duplicated('entityuen'), 'entityuen'].unique()  
r2013rest.ix[:, 'dup_ind'] = np.where(r2013rest['entityuen'].isin(dup_unique_uen), 1, 0)
#
r2013rest.ix[r2013rest.dup_ind == 1, ['entityuen', 'final_form_date']].to_excel(writer, sheet_name = "2013Q2_Q4 >= 1 RR")
r2013rest_final = r2013rest.ix[r2013rest.dup_ind == 1, :].drop_duplicates('entityuen')														#986

mi2013 = pd.concat([r2013q1_final, r2013rest_final], axis = 0, ignore_index = True)          	# there are 196 dups, that is, rated in both q1 and rest
mi2013_dedup = mi2013.sort(['entityuen', 'final_form_date']).drop_duplicates('entityuen')		#2075

#3  •	Check F2014 default list, to make sure that none of the borrowers that were flagged as non-defaults in this sample defaulted within one year of the final_form_date. 
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
df_dedup_2014 = df_dedup.ix[(df_dedup.def_date <= pd.datetime(2014, 10, 31)) & (df_dedup.def_date >= pd.datetime(2013, 11, 1)), :]

mi2013_dedup.entityuen.isin(df_dedup_2014.uen).sum()       #33, meaning 33 from mi2013_dedup are in 2014 default list, remove them
mi2013_wo_2014dft = mi2013_dedup.ix[~mi2013_dedup['entityuen'].isin(df_dedup_2014.uen), :]             #2042 left

# 4   Defaulters:
# •	Start with the list of all borrowers that have defaulted between Nov. 1, 2012 and Oct. 31, 2013. This is the total defaulted population
# •	Take the earliest available rating in F2013 data; that will be the pre-default rating associated with defaulted borrower in F2013
dft2013 = dft.ix[(dft.def_date <= pd.datetime(2013, 10, 31, 23, 59, 59)) & (dft.def_date >= pd.datetime(2012, 11, 1)), :]     
mi2013_wo_2014dft.ix[:, 'df_flag'] = np.where(mi2013_wo_2014dft.entityuen.isin(dft2013.uen), 1, 0)

# read in sic_indust info
sic_indust = pd.read_excel("H:\\work\\usgc\\2015\\quant\\SIC_Code_List.xlsx", sheetname = "sic_indust")

######################################################################################################################################################
mi2013_wo_2014dft.df_flag.value_counts()           					# {0: 2018, 1: 24}
mi2013_wo_2014dft.to_excel(writer, sheet_name = "M&I 2013 Final")
writer.save()

# US SIC   to exclude some SIC
mi2013_wo_2014dft['insudt'] = mi2013_wo_2014dft.us_sic.replace(dict(zip(sic_indust.sic_code, sic_indust.indust)))
mi2013_wo_2014dft['sector_group'] = mi2013_wo_2014dft.us_sic.replace(dict(zip(sic_indust.sic_code, sic_indust.sector_group))) 

# before exclued, industy sector
mi2013_wo_2014dft.sector_group.value_counts() 
# SRVC    451
# MFG     406
# NONP    343
# WHLS    279
# CONS    184
# REAL    105
# RETL     99
# TRAN     81
# AGRI     37
# FIN      37
# GOVT      8
# MINE      8
# FOST      4

#after exclude, industy sector counts
mi2013_after_sic = mi2013_wo_2014dft.query('sector_group not in ["NONP", "AGRI", "AGSM", "OTHR", "FOST"]')      # {1: 15, 0: 1643}

mi2013_after_sic.sector_group.value_counts() 
# SRVC    451
# MFG     406
# WHLS    279
# CONS    184
# REAL    105
# RETL     99
# TRAN     81
# FIN      37
# GOVT      8
# MINE      8

mi2013_after_sic = mi2013_after_sic.rename(columns = {'entityuen': 'uen', 'df_flag': 'default_flag'})      #  not only M&I uen {1: 15, 0: 1643},   only M&I uen {1: 14, 0: 1034}
 
print mi2013_after_sic.default_flag.value_counts(dropna = False)
 
 
#### verify
# mi2013_after_sic.ix[:, final_model_vars].count()
# cur_ast_amt               990
# tot_ast_amt               990
# cur_liab_amt              990
# tot_liab_amt              894
# tot_debt_amt              990
# net_worth_amt               0
# tangible_net_worth_amt    990
# tot_sales_amt             989
# net_inc_amt               990
# ebitda_amt                990
# dsc                       972
# yrs_in_bus                984
# debt_to_tnw_rto           983
# debt_to_ebitda_rto        983
# net_margin_rto            989
# cur_rto                   990
 

### final columns needed
common_vars = ['sk_entity_id', 'uen', 'final_form_date', 'us_sic', 'default_flag', 'default_date', 'yeartype', 'sector_group', 'quant_ranking', 'final_ranking']
final_model_vars = ['cur_ast_amt', 'tot_ast_amt', 'cur_liab_amt', 'tot_liab_amt', 'tot_debt_amt', 'net_worth_amt', 'tangible_net_worth_amt', 'tot_sales_amt', 'net_inc_amt', 'ebitda_amt', 'dsc', 'yrs_in_bus', 'debt_to_tnw_rto', 'debt_to_ebitda_rto', 'net_margin_rto', 'cur_rto']
mi2013_after_sic.ix[:, common_vars + final_model_vars].to_excel("H:\\work\\usgc\\2015\\quant\\2015_supp\\F2013MI_4_model.xlsx")

 
