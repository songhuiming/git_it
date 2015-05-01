
#!/usr/bin/python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

hgc2012vars = [u'SK_ENTITY_ID', u'EFF_DT', u'FIN_STMT_DT', 'CUR_RTO', 'TOT_AST_AMT', 'CUR_LIAB_AMT', 'TOT_LIAB_AMT', 'DEBT_TO_EBITDA_RTO', 'TANGIBLE_NET_WORTH_AMT', 'NET_SALES_AMT', 'NET_INC_AMT', 'EBITDA_AMT', 'DSC', 'INPUT_BSD', 'INPUT_EBIT']
mravars = [u'SK_ENTITY_ID', u'AS_OF_DT', u'STMT_ID', u'FIN_STMT_DT', 'CUR_AST_AMT', 'TOT_AST_AMT', 'CUR_LIAB_AMT', 'TOT_LIAB_AMT', 'TOT_DEBT_AMT', 'NET_WORTH_AMT', 'TANGIBLE_NET_WORTH_AMT', 'TOT_SALES_AMT', 'NET_INC_AMT', 'EBITDA_AMT', 'TOT_DEBT_SRVC_AMT', 'eff_date', 'EBIT_AMT', 'CASH_AND_SECU_AMT']

#read in the 2012 year start data 
hgc2012df = pd.read_excel(u"H:\\work\\usgc\\2015\\quant\\2012\\quant_2012_Year Start Data Extract_37570_V6.xlsx", sheetname = "HGC_LC")
hgc2012 = hgc2012df.ix[hgc2012df['MODEL'] == 'HGC', hgc2012vars]

#merge with population to get default flag
hgcdf = pd.read_excel(u"H:\\work\\usgc\\2015\\quant\\2012\\quant_2012_Year Start Data Extract_37570_V6.xlsx", sheetname = "populationWithDefault")
hgcdf[hgcdf.duplicated([u'BOR_SK'])]  #duplicated of 21094192

#de-dup population data for defaults
hgcdf.drop_duplicates(['BOR_SK'], inplace = True)


#read in the MRA data
#join with DCU_MRA_RISK_SPREAD_D.xls by SK_ENTITY_ID + EFF_DT(year start) to SK_ENTITY_ID +  AS_OF_DT(MAR data).
#it is unique at ['SK_ENTITY_ID', 'AS_OF_DT'] 
#mra2012 = pd.read_excel(u'H:\\work\\usgc\\2015\\quant\\DCU_MRA_RISK_SPREAD_D.xls', sheetname = "DCU_MRA_RISK_SPREAD_D")
#mra2012 = mra2012.ix[:, mravars]
#mra2012.duplicated(['SK_ENTITY_ID', 'AS_OF_DT']).sum()

#join hgc2012 with population to get default flag, but there are 44 cannot be matched
hgc2012 = pd.merge(hgc2012, hgcdf, left_on = "SK_ENTITY_ID", right_on = "BOR_SK", how = "inner")
hgc2012.DEFAULT_FLAG.value_counts(dropna = False).sort_index()  #44 cannot be matched   hgc2012.SK_ENTITY_ID[hgc2012.DEFAULT_FLAG.isnull()]

#join with MRA2012 data to get financial factors if necessary
#testjoin = pd.merge(hgc2012, mra2012, left_on=['SK_ENTITY_ID', 'EFF_DT'], right_on=['SK_ENTITY_ID', 'AS_OF_DT'], how = "left")
#a lot cannot be matched
#testjoin[['SK_ENTITY_ID', 'EFF_DT', 'AS_OF_DT']][testjoin.AS_OF_DT.isnull()].head(10)
# if match, they are the same, 95 matched, and the 95 retain_earn are the same
#(testjoin['RETAIN_EARN_AMT_y'][testjoin['RETAIN_EARN_AMT_y'].notnull()] == testjoin['RETAIN_EARN_AMT_x'][testjoin['RETAIN_EARN_AMT_y'].notnull()]).sum()
 
hgc2012['curr_assets'] = hgc2012.CUR_RTO * hgc2012.CUR_LIAB_AMT
hgc2012['total_debt'] = hgc2012.DEBT_TO_EBITDA_RTO * hgc2012.EBITDA_AMT
hgc2012['net_worth'] = hgc2012.TOT_AST_AMT - hgc2012.TOT_LIAB_AMT
hgc2012['yr_in_busi'] = pd.Series(hgc2012.EFF_DT - hgc2012.INPUT_BSD.apply(lambda x: pd.to_datetime(x))) / np.timedelta64(1, 'Y')


