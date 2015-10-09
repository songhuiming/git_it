
#!/usr/bin/python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
 
#read in the 2013 year start data  ['uen', 'naics', 'sic', 'ussic', 'dft_date', 'default_flag']  #7635
# bmo2013 = pd.read_excel(u"H:\\work\\usgc\\2015\\quant\\2013\\F2013_PD_Validation_Corp_Bank_Sov_USComm_V15.xlsx", sheetname = "sheet1")
# varbmo2013 = [x.replace(' ', '_').lower() for x in list(bmo2013)]
# bmo2013.columns = varbmo2013
# bmo2013_1 = bmo2013.ix[:, ['uen', 'naics', 'sic', 'ussic', 'dft_date', 'default_flag']]


# read in 2013 model drivers / rating data
bmo2013md = pd.read_excel(u"H:\\work\\usgc\\2015\\quant\\2013\\F2013_PD_Validation_Detailed_Model_Drivers_HGC_V2.xlsx", sheetname = "HGC")
varbmo2013md = [x.replace(' ', '_').lower() for x in list(bmo2013md)]
bmo2013md.columns = varbmo2013md

# bmo2013md  unique uen 5147
len(bmo2013md.uen.unique())
#  bmo2013md [2012-2-1, 2013-1-31]: unique uen 1532 / 1686 ratings 
len(bmo2013md.ix[(bmo2013md.final_form_date < pd.datetime(2013, 1, 31, 23, 59, 59)) & (bmo2013md.final_form_date >= pd.datetime(2012, 2, 1, 0, 0, 0))].uen.unique())
 

# bmo2013md.final_form_date <= pd.datetime(2013,1,31)  join with default after 2012, to get 21 defaults
((pd.merge(bmo2013md.ix[bmo2013md.final_form_date <= pd.datetime(2013,1,31), :].drop_duplicates('uen'), dft, left_on = 'uen' , right_on = 'UEN', how = 'left').def_date > pd.datetime(2012,11,1)) & (pd.merge(bmo2013md.ix[bmo2013md.final_form_date <= pd.datetime(2013,1,31), :].drop_duplicates('uen'), dft, left_on = 'uen' , right_on = 'UEN', how = 'left').def_date < pd.datetime(2013,10,31))).sum()

 
 
# dedup to entithy level, keep the latest EFF_DT one


#check number of defaults
 


