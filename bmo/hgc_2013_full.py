# prep of hgc2013 data (M&I will be prepared separately):
# part 1: ratings in the FACT system, join with the after2012 default list to get defaults
# part 2: ratings in HBC, 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

fact_vars = ['uen', 'entity_name', 'final_form_date', 'financialsstatdate', 'businessstartdate', 'source_from', 'us_siccode', 'currentassets', 'total_sales', 'currentliabilities', 'debt', 'tangiblenetworth', 'total_sales', 'ebitda', 'debtservicetotalamount', 'cv_debtservicecoverage_adj', 'cv_currentratio_adj', 'cv_debttotangiblenw_adj', 'cv_yearsinbusiness_adj', 'cv_debt_ebitda_adj']
hbc_vars = ['sk_entity_id', 'entity_nm', 'eff_dt', 'fin_stmt_dt', 'source_from', 'input_crnrt', 'input_cywb', 'input_dbebt', 'input_dnw', 'input_dsc', 'input_dtnw', 'input_ebiti', 'input_intr', 'input_netpr']

################################################################## Part II: ratings in HBC  ########################################################### 

# read in 2013 model drivers / rating data from HBC
#bmo2013HBC = pd.read_excel(u"H:\\work\\usgc\\2015\\quant\\2013\\2013_combined_data.xlsx", sheetname = "HBC_20110101_20131031")     #(4083, 91)
#bmo2013HBC.to_pickle(u"H:\\work\\usgc\\2015\\quant\\2013\\bmo2013HBC")
bmo2013HBC = pd.read_pickle(u"H:\\work\\usgc\\2015\\quant\\2013\\bmo2013HBC")
bmo2013HBC['source_from'] = 'HBC'
varbmo2013HBC = [x.replace(' ', '_').lower() for x in list(bmo2013HBC)]
bmo2013HBC.columns = varbmo2013HBC
# pick from 2012-2-1 to 2012-10-31, since 1)MRA data is in this interval  2)2012-11-1 to 2013-1-31 will be from FACT
bmo2013HBC_1 = bmo2013HBC.ix[(bmo2013HBC.rnd != 'Y') & (bmo2013HBC.eff_dt <= pd.datetime(2013, 1, 31)) & (bmo2013HBC.eff_dt >= pd.datetime(2012, 2, 1)), hbc_vars]
bmo2013HBC_1 = bmo2013HBC_1.rename(columns = {'entity_nm': 'entity_name', 'eff_dt': 'final_form_date'})               #  (2380, 5)

 
#read in the 2013 population data(year start data)  ['uen', 'naics', 'sic', 'ussic', 'dft_date', 'default_flag']  #7635
#bmo2013 = pd.read_excel(u"H:\\work\\usgc\\2015\\quant\\2013\\2013_combined_data.xlsx", sheetname = "2013_population")
#bmo2013.to_pickle(u"H:\\work\\usgc\\2015\\quant\\2013\\bmo2013")            # save it
bmo2013 = pd.read_pickle(u"H:\\work\\usgc\\2015\\quant\\2013\\bmo2013")
varbmo2013 = [x.replace(' ', '_').lower() for x in list(bmo2013)]
bmo2013.columns = varbmo2013

bmo2013_0 = bmo2013.query('level58_desc != "P&C CANADA" & pd_master_scale == "Commercial" & model == "General Commercial"') 
bmo2013_1 = bmo2013_0.ix[:, ['uen', 'sk_entity_id', 'ussic', 'dft_date', 'default_flag']]
bmo2013_1 = bmo2013_1.rename(columns = {'dft_date': 'default_date'}) 
bmo2013_1['default_flag'] = bmo2013_1.default_flag.replace({'Y':1, 'N': 0})
 

########################     defaults are from the year start file   ###########################
bmo2013HBC_2 = pd.merge(bmo2013HBC_1, bmo2013_1, on = 'sk_entity_id', how = 'inner') 					# left join will give same final result


# read in sic_indust info
sic_indust = pd.read_excel("H:\\work\\usgc\\2015\\quant\\SIC_Code_List.xlsx", sheetname = "sic_indust")

# to get sector_group and remove by sic
bmo2013HBC_2['indust'] = bmo2013HBC_2.ussic.replace(dict(zip(sic_indust.sic_code, sic_indust.indust)))
bmo2013HBC_2['sector_group'] = bmo2013HBC_2.ussic.replace(dict(zip(sic_indust.sic_code, sic_indust.sector_group))) 
bmo2013HBC_2_after_sic = bmo2013HBC_2.query('sector_group not in ["NONP", "AGRI", "AGSM", "OTHR", "FOST"]') 

# sort for PW
bmo2013HBC_2_after_sic.sort(['uen', 'default_flag', 'final_form_date'], ascending = [True, False, True], inplace = True)
 
#final data after sic, pw 
bmo2013HBC_2_after_sic_pw = bmo2013HBC_2_after_sic.drop_duplicates('uen') 											# {0:687, 1: 20}
bmo2013HBC_2_after_sic_pw.default_flag.value_counts()


################################################################## Part I: ratings in FACT  ###########################################################
 
# read in 2013 model drivers / rating data from FACT
#bmo2013FACT = pd.read_excel(u"H:\\work\\usgc\\2015\\quant\\2013\\2013_combined_data.xlsx", sheetname = "2013_FACT")
#bmo2013FACT.to_pickle(u"H:\\work\\usgc\\2015\\quant\\2013\\bmo2013FACT")
bmo2013FACT = pd.read_pickle(u"H:\\work\\usgc\\2015\\quant\\2013\\bmo2013FACT")
bmo2013FACT['source_from'] = 'FACT'
varbmo2013FACT = [x.replace(' ', '_').lower() for x in list(bmo2013FACT)]
bmo2013FACT.columns = varbmo2013FACT
bmo2013FACT_1 = bmo2013FACT.ix[(bmo2013FACT.rnd != 'Yes') & (bmo2013FACT.final_form_date <= pd.datetime(2013,1,31,23,59, 59)) & (bmo2013FACT.final_form_date >= pd.datetime(2012,2,1,0,0,0)), :]
bmo2013FACT_1 = bmo2013FACT_1.ix[:, fact_vars]
bmo2013FACT_1 = bmo2013FACT_1.rename(columns = {'financialsstatdate': 'fin_stmt_dt', 'us_siccode': 'ussic'})     			# 1590	
bmo2013FACT_1['businessstartdate'] = bmo2013FACT_1.businessstartdate.map(lambda x: pd.to_datetime(x))
				 

# read in M&I data
#legacy_mi = pd.read_excel(u"H:\\work\\usgc\\2015\\quant\\2013\\2013_combined_data.xlsx", sheetname = u'legacy_mi_uen_new')
#legacy_mi.to_pickle(u"H:\\work\\usgc\\2015\\quant\\2013\\mi_uen_new")
legacy_mi = pd.read_pickle(u"H:\\work\\usgc\\2015\\quant\\2013\\mi_uen_new")
legacy_mi.columns = [x.replace(' ', '_').lower() for x in list(legacy_mi)]

# remove M&I obligors
bmo2013FACT_1['mi_flag'] = bmo2013FACT_1.uen.isin(legacy_mi.uen) 
bmo2013FACT_1 = bmo2013FACT_1.query('~mi_flag')                                # 816 left, 721 unique uen

 
#################################################### read in default data ######################################################################
# read in default data and merge with quantitative data, one duplicated data was removed
#dft = pd.read_excel(u"H:\\work\\usgc\\2015\\quant\\defaults_after_Oct2012_wo_canada.xlsx", sheetname = 'df4py')
#dft.to_pickle(u"H:\\work\\usgc\\2015\\quant\\default_data")
# dft = pd.read_pickle(u"H:\\work\\usgc\\2015\\quant\\default_data")
# dft.columns = [x.lower() for x in list(dft)]

# 1 sort the default data by UEN + def_date in ascending order
# dft.sort(['uen', 'def_date'], ascending = [True, True])

# 2 check who are duplicates, the output will be  "36030947 2014-09-29"
# dft[dft.duplicated(['uen'])]

# 3 de-dup the duplicated data
# df_dedup = dft.drop_duplicates(['uen'])      # dft.groupby('uen', group_keys=False).apply(lambda x: x.ix[x.def_date.idxmax()])
# df_dedup_2013 = df_dedup.ix[(df_dedup.def_date > pd.datetime(2012,11,1)) & (df_dedup.def_date < pd.datetime(2013,10,31))]         		#393 defaults for all models in 2013_FACT
# df_dedup_2013 = df_dedup_2013.rename(columns = {'def_date': 'default_date'}) 
 
## get default information from bmo2013 = pd.read_excel(u"H:\\work\\usgc\\2015\\quant\\2013\\2013_combined_data.xlsx", sheetname = "2013_population") 
 
######################################### merge data to get default info   ################################## 
bmo2013FACT_2 = pd.merge(bmo2013FACT_1, bmo2013_1.ix[:, ['uen', u'sk_entity_id', u'default_date', u'default_flag']], on = 'uen', how = 'left')  
bmo2013FACT_2['default_flag'] = np.where(bmo2013FACT_2.default_date > 0, 1, 0) 


# read in the sic_indust data
sic_indust = pd.read_excel("H:\\work\\usgc\\2015\\quant\\SIC_Code_List.xlsx", sheetname = "sic_indust") 

bmo2013FACT_2['insudt'] = bmo2013FACT_2.ussic.replace(dict(zip(sic_indust.sic_code, sic_indust.indust)))
bmo2013FACT_2['sector_group'] = bmo2013FACT_2.ussic.replace(dict(zip(sic_indust.sic_code, sic_indust.sector_group))) 

bmo2013_FACT_after_sic = bmo2013FACT_2.query('sector_group not in ["NONP", "AGRI", "AGSM", "OTHR", "FOST"]')        	# after SIC,   left    {0:  , 1:  }

## pick FFD time interval, sort for PW purpose
bmo2013_FACT_after_sic_2 = bmo2013_FACT_after_sic.ix[(bmo2013_FACT_after_sic.final_form_date >= pd.datetime(2012,2,1,0,0,0)) & (bmo2013_FACT_after_sic.final_form_date <= pd.datetime(2013,1,31,23,59,59))]
bmo2013_FACT_after_sic_2.sort(['uen', 'default_flag', 'final_form_date'], ascending = [True, False, True], inplace = True)

bmo2013_FACT_after_sic_pw = bmo2013_FACT_after_sic_2.drop_duplicates('uen')

print bmo2013_FACT_after_sic_pw.default_flag.value_counts()           # {0: 490, 1: 3}


######################################################   Finally, Combine data and remove duplicates by sk_entity_id   ##########################################################

# after it is done, concat with bmo2013_FACT_after_sic_pw in hgc_2013_FACT.py for final dedup
bmo2013_full = pd.concat([bmo2013HBC_2_after_sic_pw, bmo2013_FACT_after_sic_pw], axis = 0)
bmo2013_full.sort(['uen', 'default_flag', 'final_form_date'], ascending = [True, False, True], inplace = True)

# finally we get 1774 obligors, compared with v15 data which gives 1619 after pw and sic
bmo2013_final = bmo2013_full.drop_duplicates('uen')  	
bmo2013_final = bmo2013_final.rename(columns = {'ussic': 'us_sic'})															# {0: 1082, 1: 22}
print bmo2013_final.default_flag.value_counts(dropna = False)

