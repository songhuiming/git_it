
#!/usr/bin/python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

hgc2012vars = [u'sk_entity_id', u'eff_dt', u'sic_code', u'fin_stmt_dt', 'cur_rto', 'tot_ast_amt', 'cur_liab_amt', 'tot_liab_amt', 'debt_to_ebitda_rto', 'tangible_net_worth_amt', 'net_sales_amt', 'net_inc_amt', 'ebitda_amt', 'dsc', 'input_bsd', 'input_ebit']
mravars = [u'sk_entity_id', u'as_of_dt', u'stmt_id', u'fin_stmt_dt', 'cur_ast_amt', 'tot_ast_amt', 'cur_liab_amt', 'tot_liab_amt', 'tot_debt_amt', 'net_worth_amt', 'tangible_net_worth_amt', 'tot_sales_amt', 'net_inc_amt', 'ebitda_amt', 'tot_debt_srvc_amt', 'eff_date', 'ebit_amt', 'cash_and_secu_amt']

#read in the 2012 year start data 
hgc2012data = pd.read_excel(u"H:\\work\\usgc\\2015\\quant\\2012\\quant_2012_Year Start Data Extract_37570_V6.xlsx", sheetname = "HGC_LC")
hgc2012data.columns = [x.replace(' ', '_').lower() for x in list(hgc2012data)]
hgc2012 = hgc2012data.ix[(hgc2012data['model'] == 'HGC') & (hgc2012data['eff_dt'] >= pd.datetime(2011,2,1,0,0,0)), hgc2012vars]
hgc2012 = hgc2012.rename(columns = {'sic_code': 'us_sic', 'eff_dt': 'final_form_date'})


#merge with population to get default flag
hgcdf = pd.read_excel(u"H:\\work\\usgc\\2015\\quant\\2012\\quant_2012_Year Start Data Extract_37570_V6.xlsx", sheetname = "populationWithDefault")
hgcdf.columns = [x.replace(' ', '_').lower() for x in list(hgcdf)]
hgcdf[hgcdf.duplicated([u'bor_sk'])]  #duplicated of 21094192
#de-dup population data for defaults
hgcdf.drop_duplicates(['bor_sk'], inplace = True)
hgcdf = hgcdf.rename(columns = {'bor_sk': 'sk_entity_id'})
hgcdf['default_flag'] = hgcdf.default_flag.replace({'Y': 1, 'N':0})

 
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

 
hgc2012_full = pd.concat([hgc2012_after_sic, bmo2012HBC_after_sic_pw], axis = 0, ignore_index = True) 
hgc2012_full.sort(['sk_entity_id', 'default_flag', 'final_form_date'], ascending = [True, False, True], inplace = True)

hgc2012_final = hgc2012_full.drop_duplicates('sk_entity_id') 
 
# hgc2012_full.drop_duplicates('sk_entity_id').default_flag.value_counts()       					

print hgc2012_final.default_flag.value_counts(dropna = False)    							# {0: 1140, 1: 43}





