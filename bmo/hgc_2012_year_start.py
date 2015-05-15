
#!/usr/bin/python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

hgc2012vars = [u'sk_entity_id', u'eff_dt', u'sic_code', u'fin_stmt_dt', 'cur_rto', 'tot_ast_amt', 'cur_liab_amt', 'tot_liab_amt', 'debt_to_ebitda_rto', 'input_dtnw', 'tangible_net_worth_amt', 'net_sales_amt', 'net_inc_amt', 'ebitda_amt', 'dsc', 'input_bsd', 'input_ebit']
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
 
 
hgc2012['cur_ast_amt'] = hgc2012.cur_rto * hgc2012.cur_liab_amt  
hgc2012['tot_debt_amt'] = hgc2012.debt_to_ebitda_rto * hgc2012.ebitda_amt   
hgc2012['net_worth_amt'] = hgc2012.tot_ast_amt - hgc2012.tot_liab_amt
hgc2012['yrs_in_bus'] = pd.Series(hgc2012.final_form_date - hgc2012.input_bsd.apply(lambda x: pd.to_datetime(x))) / np.timedelta64(1, 'Y')
# x/y: x<0 & y<0 then floor, else as is
hgc2012['net_margin_rto'] = np.where((hgc2012.net_inc_amt < 0) & (hgc2012.net_sales_amt < 0), -99999999, hgc2012.net_inc_amt / hgc2012.net_sales_amt)
 
 
# read in the sic_indust data
sic_indust = pd.read_excel("H:\\work\\usgc\\2015\\quant\\SIC_Code_List.xlsx", sheetname = "sic_indust") 

hgc2012['insudt'] = hgc2012.us_sic.replace(dict(zip(sic_indust.sic_code, sic_indust.indust)))
hgc2012['sector_group'] = hgc2012.us_sic.replace(dict(zip(sic_indust.sic_code, sic_indust.sector_group)))

outfile_12 = u"H:\\work\\usgc\\2015\\quant\\2012\\hgc2012_output_after_pw_" + pd.datetime.today().strftime("%Y_%m_%d") + '.xlsx'
writer = pd.ExcelWriter(outfile_12) 
hgc2012.to_excel(writer, sheet_name = "hgc_2012_after_pw")

# remove the not included industry sectors
hgc2012_after_sic = hgc2012.query('sector_group not in ["NONP", "AGRI", "AGSM", "OTHR", "FOST"]')  
hgc2012_after_sic.to_excel(writer, sheet_name = "hgc2012_after_sic")       			 

# some calculation
# hgc2012_after_sic['cur_ast_amt'] = hgc2012_after_sic['cur_rto'] * hgc2012_after_sic['cur_liab_amt']
# hgc2012_after_sic['net_worth_amt'] = hgc2012_after_sic['tot_ast_amt'] - hgc2012_after_sic['tot_liab_amt']  
# hgc2012_after_sic['tot_debt_amt'] = hgc2012_after_sic['debt_to_ebitda_rto'] * hgc2012_after_sic['ebitda_amt']
# hgc2012_after_sic['yrs_in_bus'] = (hgc2012_after_sic['final_form_date'] - hgc2012_after_sic['input_bsd']).apply(lambda x: pd.to_datetime(x)) / np.timedelta64(1, 'Y')


writer.save()

print hgc2012_after_sic.sector_group.value_counts()
print hgc2012_after_sic.default_flag.value_counts()
sum(hgc2012_after_sic.default_flag == 1)   		# {0: 933, 1: 42}

# there are 23 obligors with everything missing
hgc2012_after_sic.ix[hgc2012_after_sic.cur_ast_amt.isnull(), :].count()
# of which one is default
hgc2012_after_sic.ix[hgc2012_after_sic.cur_ast_amt.isnull(), :].default_flag.value_counts()

## so, the final data should be picked in this way
hgc2012_after_sic_wo_allmissing = hgc2012_after_sic.ix[hgc2012_after_sic.cur_ast_amt.notnull(), :]
hgc2012_after_sic_wo_allmissing.count() 





