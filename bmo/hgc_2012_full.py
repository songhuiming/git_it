# prep of hgc2012 data: 
# part 1: from 2011Nov to 2012Jan, the data is read in from HBC_201111_201201, then merge with population data to get defaults
# part 2: from 2011Feb to 2011Sep, data is read in from the year start file 2012_Year Start Data Extract_37570_V6.xlsx, then join with population data for defaults
# combine part 1 and part 2 data, de-dup at sk_entity_id level
# there is no data for 2011Oct because of our data collection 

#!/usr/bin/python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
######################################################   PART I: 2012 HBC data : 3 month data [2011Nov, 2012Jan]   ##########################################################
#read in the 2012 HBC ratings
#bmo2012HBC = pd.read_excel(u"H:\\work\\usgc\\2015\\quant\\2012\\2012_combined_data.xlsx", sheetname = "HBC_201111_201201")
#bmo2012HBC.to_pickle(u"H:\\work\\usgc\\2015\\quant\\2012\\2012HBC")
bmo2012HBC = pd.read_pickle(u"H:\\work\\usgc\\2015\\quant\\2012\\2012HBC")
bmo2012HBC['source_from'] = 'HBC'
varbmo2012HBC = [x.replace(' ', '_').lower() for x in list(bmo2012HBC)]
bmo2012HBC.columns = varbmo2012HBC
bmo2012HBC_1 = bmo2012HBC.ix[bmo2012HBC.rnd != 'Y', ['sk_entity_id', 'entity_nm', 'eff_dt', 'fin_stmt_dt', 'source_from', 'input_siccd']]   
bmo2012HBC_1 = bmo2012HBC_1.rename(columns = {'entity_nm': 'entity_name', 'eff_dt': 'final_form_date', 'input_siccd': 'us_sic'})


#merge with population to get default flag
#hgcdf = pd.read_excel(u"H:\\work\\usgc\\2015\\quant\\2012\\2012_combined_data.xlsx", sheetname = "populationWithDefault")
#hgcdf.to_pickle(u"H:\\work\\usgc\\2015\\quant\\2012\\hgcdf")
hgcdf = pd.read_pickle(u"H:\\work\\usgc\\2015\\quant\\2012\\hgcdf")
hgcdf.columns = [x.replace(' ', '_').lower() for x in list(hgcdf)]
hgcdf[hgcdf.duplicated([u'bor_sk'])]  #duplicated of 21094192
#de-dup population data for defaults
hgcdf.drop_duplicates(['bor_sk'], inplace = True)
hgcdf = hgcdf.rename(columns = {'bor_sk': 'sk_entity_id'})
hgcdf['default_flag'] = hgcdf.default_flag.replace({'Y': 1, 'N':0})

# read in the sic_indust data
sic_indust = pd.read_excel("H:\\work\\usgc\\2015\\quant\\SIC_Code_List.xlsx", sheetname = "sic_indust") 
 
#join 
bmo2012HBC_2 = pd.merge(bmo2012HBC_1, hgcdf, on = "sk_entity_id", how = "inner")
bmo2012HBC_2.default_flag.value_counts(dropna = False).sort_index()        		# {0: 570, 1: 38, nan: 125}, need to change to inner join
 

bmo2012HBC_2['insudt'] = bmo2012HBC_2.us_sic.replace(dict(zip(sic_indust.sic_code, sic_indust.indust)))
bmo2012HBC_2['sector_group'] = bmo2012HBC_2.us_sic.replace(dict(zip(sic_indust.sic_code, sic_indust.sector_group)))    #3251, 7312 cannot be matched

outfile_12 = u"H:\\work\\usgc\\2015\\quant\\2012\\hgc2012_output_after_pw_" + pd.datetime.today().strftime("%Y_%m_%d") + '.xlsx'
writer = pd.ExcelWriter(outfile_12) 
bmo2012HBC_2.to_excel(writer, sheet_name = "bmo2012HBC_2")

# remove the not included industry sectors
bmo2012HBC_after_sic = bmo2012HBC_2.query('sector_group in ["SRVC", "MFG", "WHLS", "CONS", "RETL", "TRAN", "FIN", "REAL", "GOVT", "MINE"]')  

#sort for dedup to sk_entity_id level
bmo2012HBC_after_sic.sort(['sk_entity_id', 'final_form_date'], inplace = True)
bmo2012HBC_after_sic_pw = bmo2012HBC_after_sic.drop_duplicates('sk_entity_id')     					# 372 in total, {0: 350, 1: 22}
bmo2012HBC_after_sic_pw.to_excel(writer, sheet_name = "bmo2012HBC_after_sic")


print bmo2012HBC_after_sic_pw.sector_group.value_counts(dropna = False)
print bmo2012HBC_after_sic_pw.default_flag.value_counts(dropna = False)   		 
 
######################################################   PART 2: 2012 YR START data : 8 month data [2011Feb, 2012Sep]   ##########################################################

hgc2012vars = [u'sk_entity_id', u'eff_dt', u'sic_code', u'fin_stmt_dt', 'cur_rto', 'tot_ast_amt', 'cur_liab_amt', 'tot_liab_amt', 'debt_to_ebitda_rto', 'tangible_net_worth_amt', 'net_sales_amt', 'net_inc_amt', 'ebitda_amt', 'dsc', 'input_bsd', 'input_ebit']
mravars = [u'sk_entity_id', u'as_of_dt', u'stmt_id', u'fin_stmt_dt', 'cur_ast_amt', 'tot_ast_amt', 'cur_liab_amt', 'tot_liab_amt', 'tot_debt_amt', 'net_worth_amt', 'tangible_net_worth_amt', 'tot_sales_amt', 'net_inc_amt', 'ebitda_amt', 'tot_debt_srvc_amt', 'eff_date', 'ebit_amt', 'cash_and_secu_amt']

#read in the 2012 year start data 
hgc2012data = pd.read_excel(u"H:\\work\\usgc\\2015\\quant\\2012\\2012_combined_data.xlsx", sheetname = "year_start")
hgc2012data.to_pickle(u"H:\\work\\usgc\\2015\\quant\\2012\\hgc2012data")
hgc2012data = pd.read_pickle(u"H:\\work\\usgc\\2015\\quant\\2012\\hgc2012data")
hgc2012data.columns = [x.replace(' ', '_').lower() for x in list(hgc2012data)]
hgc2012 = hgc2012data.ix[(hgc2012data['model'] == 'HGC') & (hgc2012data['eff_dt'] >= pd.datetime(2011,2,1,0,0,0)), hgc2012vars]
hgc2012 = hgc2012.rename(columns = {'sic_code': 'us_sic', 'eff_dt': 'final_form_date'})


# merge with population to get default flag, population default data is read in above in HBC2012 part
 
#join hgc2012 with population to get default flag, but there are some cannot be matched
hgc2012 = pd.merge(hgc2012, hgcdf, on = "sk_entity_id", how = "inner")
hgc2012.default_flag.value_counts(dropna = False).sort_index()   #  cannot be matched   hgc2012.sk_entity_id[hgc2012.default_flag.isnull()]
 
 
hgc2012['curr_assets'] = hgc2012.cur_rto * hgc2012.cur_liab_amt  
hgc2012['total_debt'] = hgc2012.debt_to_ebitda_rto * hgc2012.ebitda_amt   
hgc2012['net_worth'] = hgc2012.tot_ast_amt - hgc2012.tot_liab_amt
hgc2012['yr_in_busi'] = pd.Series(hgc2012.final_form_date - hgc2012.input_bsd.apply(lambda x: pd.to_datetime(x))) / np.timedelta64(1, 'Y')
 

hgc2012['insudt'] = hgc2012.us_sic.replace(dict(zip(sic_indust.sic_code, sic_indust.indust)))
hgc2012['sector_group'] = hgc2012.us_sic.replace(dict(zip(sic_indust.sic_code, sic_indust.sector_group)))

outfile_12 = u"H:\\work\\usgc\\2015\\quant\\2012\\hgc2012_output_after_pw_" + pd.datetime.today().strftime("%Y_%m_%d") + '.xlsx'
writer = pd.ExcelWriter(outfile_12) 
hgc2012.to_excel(writer, sheet_name = "hgc_2012_after_pw")

# remove the not included industry sectors
hgc2012_after_sic = hgc2012.query('sector_group not in ["NONP", "AGRI", "AGSM", "OTHR", "FOST"]')  
hgc2012_after_sic.to_excel(writer, sheet_name = "hgc2012_after_sic")       			 

writer.save()

print hgc2012_after_sic.sector_group.value_counts()
print hgc2012_after_sic.default_flag.value_counts()
sum(hgc2012_after_sic.default_flag == 1)   		# {0: 933, 1: 42}

######################################################   Finally, Combine data and remove duplicates by sk_entity_id   ##########################################################
 
hgc2012_full = pd.concat([hgc2012_after_sic, bmo2012HBC_after_sic_pw], axis = 0, ignore_index = True) 
hgc2012_full.sort(['sk_entity_id', 'default_flag', 'final_form_date'], ascending = [True, False, True], inplace = True)

hgc2012_final = hgc2012_full.drop_duplicates('sk_entity_id') 
 
# hgc2012_full.drop_duplicates('sk_entity_id').default_flag.value_counts()       					

print hgc2012_final.default_flag.value_counts(dropna = False)    							# {0: 1140, 1: 43}




