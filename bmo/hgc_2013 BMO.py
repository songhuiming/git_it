
#!/usr/bin/python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
 
#read in the 2013 year start data  ['uen', 'naics', 'sic', 'ussic', 'dft_date', 'default_flag']  #7635
bmo2013 = pd.read_excel(u"H:\\work\\usgc\\2015\\quant\\2013\\F2013_PD_Validation_Corp_Bank_Sov_USComm_V15.xlsx", sheetname = "Sheet1")
varbmo2013 = [x.replace(' ', '_').lower() for x in list(bmo2013)]
bmo2013.columns = varbmo2013
bmo2013_1 = bmo2013.ix[:, ['uen', 'naics', 'sic', 'ussic', 'dft_date', 'default_flag']]
bmo2013_0 = bmo2013.query('level58_desc != "P&C Canada" & pd_master_scale == "Commercial" & model == "General Commercial"') 


# read in 2013 model drivers / rating data
bmo2013md = pd.read_excel(u"H:\\work\\usgc\\2015\\quant\\2013\\F2013_PD_Validation_Detailed_Model_Drivers_HGC_V2.xlsx", sheetname = "HGC")
varbmo2013md = [x.replace(' ', '_').lower() for x in list(bmo2013md)]
bmo2013md.columns = varbmo2013md

# bmo2013md  unique uen 5147
len(bmo2013md.uen.unique())

#  bmo2013md [2012-2-1, 2013-1-31]: unique uen 1532 / 1686 ratings 
bmo2013 = bmo2013md.ix[(bmo2013md.final_form_date < pd.datetime(2013, 1, 31, 23, 59, 59)) & (bmo2013md.final_form_date >= pd.datetime(2012, 2, 1, 0, 0, 0))]
 
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
df_dedup_2013 = df_dedup.ix[(df_dedup.def_date > pd.datetime(2012,11,1)) & (df_dedup.def_date < pd.datetime(2013,10,31))] 
#################################################################################################################################################


# bmo2013md.final_form_date <= pd.datetime(2013,1,31)  join with default after 2012, to get 22 defaults
bmo2013_final = pd.merge(bmo2013, df_dedup_2013, on = 'uen', how = 'left').sort(['uen', 'final_form_date']).drop_duplicates('uen') 
bmo2013_final['df_flag'] = np.where(bmo2013_final.def_date > 0, 1, 0)        	 # bmo2013_final.df_flag.value_counts() {0: 1510, 1: 22}
bmo2013_final.df_flag.value_counts(dropna = False).sort_index()
 
#US_SICCode
bmo2013_final['insudt'] = bmo2013_final.us_siccode.replace(dict(zip(sic_indust.sic_code, sic_indust.indust)))
bmo2013_final['sector_group'] = bmo2013_final.us_siccode.replace(dict(zip(sic_indust.sic_code, sic_indust.sector_group))) 
 
 # Exclude some industry factors
bmo2013_after_sic = bmo2013_final.query('sector_group not in ["NONP", "AGRI", "AGSM", "OTHR", "FOST"]')      # {0: 1169, 1: 14}

bmo2013_after_sic.sector_group.value_counts()
# SRVC    341
# MFG     270
# WHLS    192
# CONS    122
# REAL     75
# TRAN     69
# RETL     69
# FIN      27
# GOVT     13
# MINE      5


