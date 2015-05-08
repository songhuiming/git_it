
#!/usr/bin/python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

hgc2012vars = [u'sk_entity_id', u'eff_dt', u'sic_code', u'fin_stmt_dt', 'cur_rto', 'tot_ast_amt', 'cur_liab_amt', 'tot_liab_amt', 'debt_to_ebitda_rto', 'tangible_net_worth_amt', 'net_sales_amt', 'net_inc_amt', 'ebitda_amt', 'dsc', 'input_bsd', 'input_ebit']
mravars = [u'sk_entity_id', u'as_of_dt', u'stmt_id', u'fin_stmt_dt', 'cur_ast_amt', 'tot_ast_amt', 'cur_liab_amt', 'tot_liab_amt', 'tot_debt_amt', 'net_worth_amt', 'tangible_net_worth_amt', 'tot_sales_amt', 'net_inc_amt', 'ebitda_amt', 'tot_debt_srvc_amt', 'eff_date', 'ebit_amt', 'cash_and_secu_amt']

#read in the 2012 year start data 
hgc2012data = pd.read_excel(u"H:\\work\\usgc\\2015\\quant\\2012\\quant_2012_Year Start Data Extract_37570_V6.xlsx", sheetname = "HGC_LC")
hgc2012data.columns = [x.replace(' ', '_').lower() for x in list(hgc2012data)]

hgc2012 = hgc2012data.ix[hgc2012data['model'] == 'HGC', hgc2012vars]


#merge with population to get default flag
hgcdf = pd.read_excel(u"H:\\work\\usgc\\2015\\quant\\2012\\quant_2012_Year Start Data Extract_37570_V6.xlsx", sheetname = "populationWithDefault")
hgcdf.columns = [x.replace(' ', '_').lower() for x in list(hgcdf)]
hgcdf[hgcdf.duplicated([u'bor_sk'])]  #duplicated of 21094192

#de-dup population data for defaults
hgcdf.drop_duplicates(['bor_sk'], inplace = True)


#read in the MRA data
#join with DCU_MRA_RISK_SPREAD_D.xls by sk_entity_id + EFF_DT(year start) to sk_entity_id +  AS_OF_DT(MAR data).
#it is unique at ['sk_entity_id', 'AS_OF_DT'] 
#mra2012 = pd.read_excel(u'H:\\work\\usgc\\2015\\quant\\DCU_MRA_RISK_SPREAD_D.xls', sheetname = "DCU_MRA_RISK_SPREAD_D")
#mra2012 = mra2012.ix[:, mravars]
#mra2012.duplicated(['sk_entity_id', 'AS_OF_DT']).sum()

#join hgc2012 with population to get default flag, but there are 44 cannot be matched
hgc2012 = pd.merge(hgc2012, hgcdf, left_on = "sk_entity_id", right_on = "bor_sk", how = "inner")
hgc2012.default_flag.value_counts(dropna = False).sort_index()  #44 cannot be matched   hgc2012.sk_entity_id[hgc2012.default_flag.isnull()]

#join with MRA2012 data to get financial factors if necessary
#testjoin = pd.merge(hgc2012, mra2012, left_on=['sk_entity_id', 'EFF_DT'], right_on=['sk_entity_id', 'AS_OF_DT'], how = "left")
#a lot cannot be matched
#testjoin[['sk_entity_id', 'EFF_DT', 'AS_OF_DT']][testjoin.AS_OF_DT.isnull()].head(10)
# if match, they are the same, 95 matched, and the 95 retain_earn are the same
#(testjoin['RETAIN_EARN_AMT_y'][testjoin['RETAIN_EARN_AMT_y'].notnull()] == testjoin['RETAIN_EARN_AMT_x'][testjoin['RETAIN_EARN_AMT_y'].notnull()]).sum()
 
hgc2012['curr_assets'] = hgc2012.cur_rto * hgc2012.cur_liab_amt  
hgc2012['total_debt'] = hgc2012.debt_to_ebitda_rto * hgc2012.ebitda_amt   
hgc2012['net_worth'] = hgc2012.tot_ast_amt - hgc2012.tot_liab_amt
hgc2012['yr_in_busi'] = pd.Series(hgc2012.eff_dt - hgc2012.input_bsd.apply(lambda x: pd.to_datetime(x))) / np.timedelta64(1, 'Y')

[x for x in list(hgc2012data) if 'sic' in x]

hgc2012 = hgc2012.rename(columns = {'sic_code': 'us_sic', 'eff_dt': 'final_form_date'})

hgc2012['insudt'] = hgc2012.us_sic.replace(dict(zip(sic_indust.sic_code, sic_indust.indust)))
hgc2012['sector_group'] = hgc2012.us_sic.replace(dict(zip(sic_indust.sic_code, sic_indust.sector_group)))

outfile_12 = u"H:\\work\\usgc\\2015\\quant\\2012\\hgc2012_output_after_pw_" + pd.datetime.today().strftime("%Y_%m_%d") + '.xlsx'
writer = pd.ExcelWriter(outfile_12) 
hgc2012.to_excel(writer, sheet_name = "hgc_2012_after_pw")

# remove the not included industry sectors
hgc2012_after_sic = hgc2012.query('sector_group not in ["NONP", "AGRI", "AGSM", "OTHR", "FOST"]')  #hgc2012.ix[~hgc2012.sector_group.isin(['NONP', 'AGRI', 'AGSM', 'OTHR', 'FOST'])]
hgc2012_after_sic.to_excel(writer, sheet_name = "hgc2012_after_sic")

writer.save()

hgc2012_after_sic.sector_group.value_counts()
sum(hgc2012_after_sic.default_flag == 'Y')   		#45 defaults, 1167 non-defaults


#  number of non-defaults to be sampled     

 





