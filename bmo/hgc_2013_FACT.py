##  FACT data, get defaults from the default list after 2012


#!/usr/bin/python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
# read in 2013 model drivers / rating data from FACT
#bmo2013FACT = pd.read_excel(u"H:\\work\\usgc\\2015\\quant\\2013\\2013_combined_data.xlsx", sheetname = "2013_FACT")
#bmo2013FACT.to_pickle(u"H:\\work\\usgc\\2015\\quant\\2013\\bmo2013FACT")
bmo2013FACT = pd.read_pickle(u"H:\\work\\usgc\\2015\\quant\\2013\\bmo2013FACT")
bmo2013FACT['source_from'] = 'FACT'
varbmo2013FACT = [x.replace(' ', '_').lower() for x in list(bmo2013FACT)]
bmo2013FACT.columns = varbmo2013FACT
bmo2013FACT_1 = bmo2013FACT.ix[(bmo2013FACT.rnd != 'Yes') & (bmo2013FACT.final_form_date <= pd.datetime(2013,1,31,23,59, 59)) & (bmo2013FACT.final_form_date >= pd.datetime(2012,2,1,0,0,0)), ['uen', 'entity_name', 'final_form_date', 'financialsstatdate', 'source_from', 'us_siccode']]
bmo2013FACT_1 = bmo2013FACT_1.rename(columns = {'financialsstatdate': 'fin_stmt_dt', 'us_siccode': 'ussic'})     			# 1590					 

# read in M&I data
legacy_mi = pd.read_excel(u"H:\\work\\usgc\\2015\\quant\\2013\\2013_combined_data.xlsx", sheetname = u'Legacy MI UEN')
legacy_mi.columns = [x.replace(' ', '_').lower() for x in list(legacy_mi)]

# remove M&I obligors
bmo2013FACT_1['mi_flag'] = bmo2013FACT_1.uen.isin(legacy_mi.uen) 
bmo2013FACT_1 = bmo2013FACT_1.query('~mi_flag')                                # 789 left, 696 unique uen

 
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

bmo2013_FACT_after_sic_pw.default_flag.value_counts()                                           # {0: 470,  1: 1} 
 

