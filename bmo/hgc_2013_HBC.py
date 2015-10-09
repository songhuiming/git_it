# use HBC_20110101_20131031 for data preparation, the only way to get default if from the population data v15 excel file


#!/usr/bin/python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
 
#read in the 2013 year start data  ['uen', 'naics', 'sic', 'ussic', 'dft_date', 'default_flag']  #7635
bmo2013 = pd.read_excel(u"H:\\work\\usgc\\2015\\quant\\2013\\2013_combined_data.xlsx", sheetname = "2013_population")
varbmo2013 = [x.replace(' ', '_').lower() for x in list(bmo2013)]
bmo2013.columns = varbmo2013

bmo2013_0 = bmo2013.query('level58_desc != "P&C CANADA" & pd_master_scale == "Commercial" & model == "General Commercial"') 
bmo2013_1 = bmo2013_0.ix[:, ['uen', 'sk_entity_id', 'ussic', 'dft_date', 'default_flag']]
bmo2013_1['default_flag'] = bmo2013_1.default_flag.replace({'Y':1, 'N': 0})
 
 
# read in 2013 model drivers / rating data from HBC
bmo2013HBC = pd.read_excel(u"H:\\work\\usgc\\2015\\quant\\2013\\2013_combined_data.xlsx", sheetname = "HBC_20110101_20131031")
bmo2013HBC['source_from'] = 'HBC'
varbmo2013HBC = [x.replace(' ', '_').lower() for x in list(bmo2013HBC)]
bmo2013HBC.columns = varbmo2013HBC
bmo2013HBC_1 = bmo2013HBC.ix[bmo2013HBC.rnd != 'Y', ['sk_entity_id', 'entity_nm', 'eff_dt', 'fin_stmt_dt', 'source_from']]
bmo2013HBC_1 = bmo2013HBC_1.rename(columns = {'entity_nm': 'entity_name', 'eff_dt': 'final_form_date'})


########################    ?? How to get default for this?   ###########################
bmo2013HBC_2 = pd.merge(bmo2013HBC_1, bmo2013_1, on = 'sk_entity_id', how = 'inner')

# to get sector_group and remove by sic
bmo2013HBC_2['insudt'] = bmo2013HBC_2.ussic.replace(dict(zip(sic_indust.sic_code, sic_indust.indust)))
bmo2013HBC_2['sector_group'] = bmo2013HBC_2.ussic.replace(dict(zip(sic_indust.sic_code, sic_indust.sector_group))) 
bmo2013HBC_2_after_sic = bmo2013HBC_2.query('sector_group not in ["NONP", "AGRI", "AGSM", "OTHR", "FOST"]') 

# sort for PW
bmo2013HBC_2_after_sic.sort(['uen', 'default_flag', 'final_form_date'], ascending = [True, False, True], inplace = True)
 
#final data after sic, pw 
bmo2013HBC_2_after_sic_pw = bmo2013HBC_2_after_sic.drop_duplicates('uen') 											# {0:789, 1: 21}


#######################################################   STOP HERE  #############################################################



# after it is done, concat with bmo2013_FACT_after_sic_pw in hgc_2013_FACT.py for final dedup
bmo2013_all = pd.concat([bmo2013HBC_2_after_sic_pw, bmo2013_FACT_after_sic_pw], axis = 0)
bmo2013_all.sort(['uen', 'default_flag', 'final_form_date'], ascending = [True, False, True], inplace = True)

# finally we get 1774 obligors, compared with v15 data which gives 1619 after pw and sic
bmo2013_final = bmo2013_all.drop_duplicates('uen')  																# {0: 1740, 1: 34}
print bmo2013_final.default_flag.value_counts(dropna = False)





