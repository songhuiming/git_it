##  FACT data, get defaults from the default list after 2012


#!/usr/bin/python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pdIntval = [0.0003, 0.0007, 0.0011, 0.0019, 0.0032, 0.0054, 0.0091, 0.0154, 0.0274, 0.0516, 0.097, 0.1823, 1]
msRating = ['I-2', 'I-3', 'I-4', 'I-5', 'I-6', 'I-7', 'S-1', 'S-2', 'S-3', 'S-4', 'P-1', 'P-2', 'P-3', 'D-1']
pcRR = [10, 15, 20, 25, 30, 35, 40, 45, 46, 47, 50, 51, 52]
ranking = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
newMapPD = [0.0003, 0.0005, 0.0009, 0.0015, 0.0026, 0.0043, 0.0073, 0.0123, 0.0214, 0.0395, 0.0743, 0.1397, 0.2412]
oldMapPD = [0.0004, 0.0007, 0.0013, 0.0025, 0.004, 0.0067, 0.0107, 0.018, 0.0293, 0.0467, 0.0933, 0.18, 0.3333]
 
# read in 2013 model drivers / rating data from FACT
#bmo2013FACT = pd.read_excel(u"H:\\work\\usgc\\2015\\quant\\2013\\2013_combined_data.xlsx", sheetname = "2013_FACT")
#bmo2013FACT.to_pickle(u"H:\\work\\usgc\\2015\\quant\\2013\\bmo2013FACT")

bmo2013FACT = pd.read_pickle(u"H:\\work\\usgc\\2015\\quant\\2013\\bmo2013FACT")
bmo2013FACT['source_from'] = 'FACT'
varbmo2013FACT = [x.replace(' ', '_').lower() for x in list(bmo2013FACT)]
bmo2013FACT.columns = varbmo2013FACT

bmo2013FACT['quant_ranking'] = bmo2013FACT.quantitative_rating.map(dict(zip(msRating, ranking)))
bmo2013FACT['final_ranking'] = bmo2013FACT.final_rating.map(dict(zip(msRating, ranking)))

bmo2013FACT_1 = bmo2013FACT.ix[(bmo2013FACT.rnd != 'Yes') & (bmo2013FACT.final_form_date <= pd.datetime(2013,1,31,23,59, 59)) & (bmo2013FACT.final_form_date >= pd.datetime(2012,2,1,0,0,0)), :]
bmo2013FACT_1 = bmo2013FACT_1.rename(columns = {'financialsstatdate': 'fin_stmt_dt', 'us_siccode': 'ussic'})     			# 1590					 
bmo2013FACT_1['businessstartdate'] = bmo2013FACT_1['businessstartdate'].apply(lambda x: pd.to_datetime(x))

# read in M&I data
legacy_mi = pd.read_excel(u"H:\\work\\usgc\\2015\\quant\\2013\\2013_combined_data.xlsx", sheetname = u'legacy_mi_uen_new')
legacy_mi.columns = [x.replace(' ', '_').lower() for x in list(legacy_mi)]

# remove M&I obligors
bmo2013FACT_1['mi_flag'] = bmo2013FACT_1.uen.isin(legacy_mi.uen) 
bmo2013FACT_1 = bmo2013FACT_1.query('~mi_flag')                                # 789 left, 696 unique uen

#################################################### read in supplementary data ######################################################################
F2013FACT_sup_cols = ['archive_statement_id', 'archiveid', 'totalliabilitiesexcldeferredincometax', 'netincome']  #'totalassets','debt', 'debtservicecoverage'
#f2013fact_sup_data = pd.read_excel("H:\\work\\usgc\\2015\\quant\\2015_supp\\HGC_FACT_Financial_Variable_V1.xlsx")
#f2013fact_sup_data.columns = [x.replace(' ', '_').lower() for x in list(f2013fact_sup_data)]
#f2013fact_sup_data.to_pickle("H:\\work\\usgc\\2015\\quant\\2015_supp\\f2013fact_sup_dat")
f2013fact_sup_data = pd.read_pickle("H:\\work\\usgc\\2015\\quant\\2015_supp\\f2013fact_sup_dat")
f2013fact_sup_data_1 = f2013fact_sup_data.ix[:, F2013FACT_sup_cols]
F2013FACT_sup_cols_rename = {'totalliabilitiesexcldeferredincometax': 'tot_liab_amt', 'netincome': 'net_inc_amt'}   #'totalassets': 'tot_ast_amt',  'debt': 'tot_debt_amt', 'debtservicecoverage': 'dsc'
f2013fact_sup_data_1 = f2013fact_sup_data_1.rename(columns = F2013FACT_sup_cols_rename)

bmo2013FACT_1 = pd.merge(bmo2013FACT_1, f2013fact_sup_data_1, on = 'archive_statement_id', how = 'left') 
 
#################################################### read in default data ######################################################################
 
## get default information from bmo2013 = pd.read_excel(u"H:\\work\\usgc\\2015\\quant\\2013\\2013_combined_data.xlsx", sheetname = "2013_population") 
bmo2013 = pd.read_pickle(u"H:\\work\\usgc\\2015\\quant\\2013\\bmo2013")
varbmo2013 = [x.replace(' ', '_').lower() for x in list(bmo2013)]
bmo2013.columns = varbmo2013

bmo2013_0 = bmo2013.query('level58_desc != "P&C CANADA" & pd_master_scale == "Commercial" & model == "General Commercial"') 
bmo2013_1 = bmo2013_0.ix[:, ['uen', 'sk_entity_id', 'ussic', 'dft_date', 'default_flag']]
bmo2013_1 = bmo2013_1.rename(columns = {'dft_date': 'default_date'}) 
bmo2013_1['default_flag'] = bmo2013_1.default_flag.replace({'Y':1, 'N': 0})
 
######################################### merge data to get default info   ################################## 
bmo2013FACT_2 = pd.merge(bmo2013FACT_1, bmo2013_1.ix[:, ['uen', u'sk_entity_id', u'default_date', u'default_flag']], on = 'uen', how = 'left')  
bmo2013FACT_2['default_flag'] = np.where(bmo2013FACT_2.default_date > 0, 1, 0) 


# read in the sic_indust data
sic_indust = pd.read_excel("H:\\work\\usgc\\2015\\quant\\SIC_Code_List.xlsx", sheetname = "sic_indust") 

bmo2013FACT_2['indust'] = bmo2013FACT_2.ussic.replace(dict(zip(sic_indust.sic_code, sic_indust.indust)))
bmo2013FACT_2['sector_group'] = bmo2013FACT_2.ussic.replace(dict(zip(sic_indust.sic_code, sic_indust.sector_group))) 

bmo2013_FACT_after_sic = bmo2013FACT_2.query('sector_group not in ["NONP", "AGRI", "AGSM", "OTHR", "FOST"]')        	# after SIC,   left    {0:  , 1:  }

## pick FFD time interval, sort for PW purpose
bmo2013_FACT_after_sic_2 = bmo2013_FACT_after_sic.ix[(bmo2013_FACT_after_sic.final_form_date >= pd.datetime(2012,2,1,0,0,0)) & (bmo2013_FACT_after_sic.final_form_date <= pd.datetime(2013,1,31,23,59,59))]
bmo2013_FACT_after_sic_2.sort(['uen', 'default_flag', 'final_form_date'], ascending = [True, False, True], inplace = True)

bmo2013_FACT_after_sic_pw = bmo2013_FACT_after_sic_2.drop_duplicates('uen')

bmo2013_FACT_after_sic_pw.default_flag.value_counts()                                           # {0: 470,  1: 1} 
f2013_fact_rename1 = {"cv_debtservicecoverage_adj": "dsc", "cv_currentratio_adj": "cur_rto_1", "cv_debttotangiblenw_adj": "debt_to_tnw_rto", "cv_yearsinbusiness_adj": "yib_missing", "total_sales": "tot_sales", "cv_debt_ebitda_adj": "debt_to_ebitda_rto"}
f2013_fact_rename2 = {'businessstartdate': 'bsd', 'ussic': 'us_sic', 'total_assets': 'tot_ast_amt', 'currentassets': 'cur_ast_amt', 'tot_sales': 'tot_sales_amt', 'currentliabilities': 'cur_liab_amt', 'debt': 'tot_debt_amt', 'tangiblenetworth': 'tangible_net_worth_amt', 'ebitda': 'ebitda_amt', 'debtservicetotalamount': 'ds_amt', 'debt_to_tnw': 'debt_to_tnw_rto', 'debt_to_ebitda': 'debt_to_ebitda_rto'}
bmo2013_FACT_after_sic_pw = bmo2013_FACT_after_sic_pw.rename(columns = f2013_fact_rename1)
bmo2013_FACT_after_sic_pw = bmo2013_FACT_after_sic_pw.rename(columns = f2013_fact_rename2)

## table19_calc:  Non-Debt Based Ratios ( except Net Margin, EBITDA Margin, EBIT Margin)
def table19_calc(x, y):
	return np.where((x.isnull()) | (y.isnull()), np.nan, np.where((x == 0) & (y == 0), 0, np.where((x > 0) & (y == 0), 99999999, np.where((x < 0) & (y == 0), -99999999, np.where((x < 0) & (y < 0), -99999999, x / y )))))
 
# years in business 	
def yib_calc(x, y):
	return (x - y) / np.timedelta64(1, 'Y')
	
# cur_rto = cur_ast_amt/cur_liab_amt:   							# table 19
# debt_to_tnw_rto = tot_debt_amt / tangible_net_worth_amt			# table 19
# yrs_in_bus  =  final_form_date - bsd								# direct calc
# net_margin_rto =  net_inc_amt / tot_sales_amt						# table 19
 
	
bmo2013_FACT_after_sic_pw['yrs_in_bus'] = yib_calc(bmo2013_FACT_after_sic_pw['final_form_date'], bmo2013_FACT_after_sic_pw['bsd'])
bmo2013_FACT_after_sic_pw['cur_rto'] = table19_calc(bmo2013_FACT_after_sic_pw['cur_ast_amt'], bmo2013_FACT_after_sic_pw['cur_liab_amt'])  
bmo2013_FACT_after_sic_pw['net_margin_rto'] = table19_calc(bmo2013_FACT_after_sic_pw['net_inc_amt'], bmo2013_FACT_after_sic_pw['tot_sales_amt'])     #need to pull net_inc_amt first
bmo2013_FACT_after_sic_pw['yeartype'] = '2013HGC'
	

