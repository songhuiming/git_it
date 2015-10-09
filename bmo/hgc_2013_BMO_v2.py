# update: this method doesnt work because a lot of data will be filtered out.

## use the following way to prepare data
# 1. Use population file as the basis  (V15)
# 2. Get the SK to UEN mapping from Zack
# 3. Get drivers from FACT driver file based on UEN matching (v2)
# 4. Get drivers from HBC file based on SK to UEN map provided by Zack (HBC_20110101_20131031.xlsx)

# doesnt woek: F2013_PD_Validation_Corp_Bank_Sov_USComm_V15.xlsx year start(or call populaiton) data only have ratings end at 2012Oct. we need ratings until 2013Jan.


#!/usr/bin/python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
 
#read in the 2013 year start data  ['uen', 'naics', 'sic', 'ussic', 'dft_date', 'default_flag']  #7635
bmo2013 = pd.read_excel(u"H:\\work\\usgc\\2015\\quant\\2013\\2013_combined_data.xlsx", sheetname = "2013_population")
varbmo2013 = [x.replace(' ', '_').lower() for x in list(bmo2013)]
bmo2013.columns = varbmo2013
bmo2013_0 = bmo2013.query('level58_desc != "P&C CANADA" & pd_master_scale == "Commercial" & model == "General Commercial"') 
bmo2013_1 = bmo2013_0.ix[:, ['uen', 'sk_entity_id', 'ussic', 'dft_date', 'default_flag', 'dcu_eff_date', 'fact_ff_date']]
#bmo2013_1 = bmo2013_1.rename(columns = {'fact_ff_date': 'final_form_date'})
bmo2013_1['default_date'] = pd.to_datetime(bmo2013_1['dft_date'])
bmo2013_1['eff_dt'] = pd.to_datetime(bmo2013_1['dcu_eff_date'])
bmo2013_1['final_form_date'] = np.where(bmo2013_1['fact_ff_date'] > 0, bmo2013_1['fact_ff_date'], bmo2013_1['eff_dt'])

#read in Legacy M&I New Uen
legacy_mi = pd.read_excel(u"H:\\work\\usgc\\2015\\quant\\2013\\2013_combined_data.xlsx", sheetname = u'legacy_mi_uen_new')
bmo2013_1 = bmo2013_1.ix[~bmo2013_1['uen'].isin(legacy_mi.uen), :]

# read in 2013 model drivers / rating data from FACT
bmo2013FACT = pd.read_excel(u"H:\\work\\usgc\\2015\\quant\\2013\\2013_combined_data.xlsx", sheetname = "2013_FACT")
bmo2013FACT['source_from'] = 'FACT'
varbmo2013FACT = [x.replace(' ', '_').lower() for x in list(bmo2013FACT)]
bmo2013FACT.columns = varbmo2013FACT
bmo2013FACT_1 = bmo2013FACT.ix[bmo2013FACT.rnd != 'Yes', ['uen', 'entity_name', 'final_form_date', 'financialsstatdate', 'source_from']]
bmo2013FACT_1['final_form_date'] = map(lambda x: pd.to_datetime(x.date()), bmo2013FACT_1['final_form_date'])
bmo2013FACT_1 = bmo2013FACT_1.rename(columns = {'financialsstatdate': 'fin_stmt_dt'})

# read in 2013 model drivers / rating data from HBC
bmo2013HBC = pd.read_excel(u"H:\\work\\usgc\\2015\\quant\\2013\\2013_combined_data.xlsx", sheetname = "HBC_20110101_20131031")
bmo2013HBC['source_from'] = 'HBC'
varbmo2013HBC = [x.replace(' ', '_').lower() for x in list(bmo2013HBC)]
bmo2013HBC.columns = varbmo2013HBC
bmo2013HBC_1 = bmo2013HBC.ix[bmo2013HBC.rnd != 'Y', ['sk_entity_id', 'entity_nm', 'eff_dt', 'fin_stmt_dt', 'source_from']]
bmo2013HBC_1 = bmo2013HBC_1.rename(columns = {'entity_nm': 'entity_name'})


########     .ix[bmo2013_1_1.final_form_date.isnull(), ['uen', 'sk_entity_id', 'ussic', 'dft_date', 'default_flag', 'eff_dt']]
bmo2013_1_1 = pd.merge(bmo2013_1, bmo2013FACT_1, on = ['uen', 'final_form_date'], how = 'inner')
bmo2013_1_1['priority'] = 1
bmo2013_1_2 = pd.merge(bmo2013_1, bmo2013HBC_1, on = ['sk_entity_id', 'eff_dt'], how = 'inner')
bmo2013_1_2['priority'] = 2
bmo2013_1_3 = bmo2013_1.ix[~(bmo2013_1.uen.isin(bmo2013_1_1.uen)) & ~(bmo2013_1.uen.isin(bmo2013_1_2.uen))]
bmo2013_1_3['priority'] = 3

bmo2013_concat = pd.concat([bmo2013_1_1, bmo2013_1_2, bmo2013_1_3], axis = 0)
bmo2013_concat.sort(['uen', 'priority'])
bmo2013_concat = bmo2013_concat.drop_duplicates('uen')
bmo2013_concat.ix[bmo2013_concat.source_from.notnull()].shape

# SIC filter
sic_indust = pd.read_excel("H:\\work\\usgc\\2015\\quant\\SIC_Code_List.xlsx", sheetname = "sic_indust") 
bmo2013_concat['insudt'] = bmo2013_concat.ussic.replace(dict(zip(sic_indust.sic_code, sic_indust.indust)))
bmo2013_concat['sector_group'] = bmo2013_concat.ussic.replace(dict(zip(sic_indust.sic_code, sic_indust.sector_group))) 

bmo2013_concat_after_sic = bmo2013_concat.query('sector_group not in ["NONP", "AGRI", "AGSM", "OTHR", "FOST"]')        	# after SIC, 1619 left    {0: 1593, 1: 26}
# bmo2013_concat_after_sic.ix[bmo2013_concat_after_sic.source_from.notnull()]												# after rmv mis FFD, 1024 left,  {0: 1001, 1: 23}
bmo2013_concat_after_sic_pw = bmo2013_concat_after_sic.ix[(bmo2013_concat_after_sic.final_form_date >= pd.datetime(2012,2,1,0,0,0)) &(bmo2013_concat_after_sic.final_form_date <= pd.datetime(2013,1,31,23,59,59))]
bmo2013_concat_after_sic_pw.drop_duplicates('uen').default_flag.value_counts()                                   		# after pw 473 left,  {0:455, 1:18}				


 

