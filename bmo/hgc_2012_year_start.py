
#!/usr/bin/python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

hgc2012vars = [u'sk_entity_id', u'eff_dt', u'sic_code', u'fin_stmt_dt', 'quan_rr', 'risk_rtg', 'cur_rto', 'tot_ast_amt', 'cur_liab_amt', 'tot_liab_amt', 'debt_to_ebitda_rto', 'input_dtnw', 'tangible_net_worth_amt', 'net_sales_amt', 'net_inc_amt', 'ebitda_amt', 'dsc', 'input_bsd', 'input_ebit']
mravars = [u'sk_entity_id', u'as_of_dt', u'stmt_id', u'fin_stmt_dt', 'cur_ast_amt', 'tot_ast_amt', 'cur_liab_amt', 'tot_liab_amt', 'tot_debt_amt', 'net_worth_amt', 'tangible_net_worth_amt', 'tot_sales_amt', 'net_inc_amt', 'ebitda_amt', 'tot_debt_srvc_amt', 'eff_date', 'ebit_amt', 'cash_and_secu_amt']

#read in the 2012 year start data 
hgc2012data = pd.read_excel(u"H:\\work\\usgc\\2015\\quant\\2012\\quant_2012_Year Start Data Extract_37570_V6.xlsx", sheetname = "HGC_LC")
hgc2012data.columns = [x.replace(' ', '_').lower() for x in list(hgc2012data)]
hgc2012 = hgc2012data.ix[(hgc2012data['model'] == 'HGC') & (hgc2012data['eff_dt'] >= pd.datetime(2011,2,1,0,0,0)), hgc2012vars]
hgc2012 = hgc2012.rename(columns = {'sic_code': 'us_sic', 'eff_dt': 'final_form_date'})

pdIntval = [0.0003, 0.0007, 0.0011, 0.0019, 0.0032, 0.0054, 0.0091, 0.0154, 0.0274, 0.0516, 0.097, 0.1823, 1]
msRating = ['I-2', 'I-3', 'I-4', 'I-5', 'I-6', 'I-7', 'S-1', 'S-2', 'S-3', 'S-4', 'P-1', 'P-2', 'P-3', 'D-1']
pcRR = [10, 15, 20, 25, 30, 35, 40, 45, 46, 47, 50, 51, 52]
ranking = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
newMapPD = [0.0003, 0.0005, 0.0009, 0.0015, 0.0026, 0.0043, 0.0073, 0.0123, 0.0214, 0.0395, 0.0743, 0.1397, 0.2412]
oldMapPD = [0.0004, 0.0007, 0.0013, 0.0025, 0.004, 0.0067, 0.0107, 0.018, 0.0293, 0.0467, 0.0933, 0.18, 0.3333]

hgc2012['quant_ranking'] = hgc2012.quan_rr.map(dict(zip(pcRR, ranking)))
hgc2012['final_ranking'] = hgc2012.risk_rtg.map(dict(zip(pcRR, ranking)))

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
 
## table19_calc:  Non-Debt Based Ratios ( except Net Margin, EBITDA Margin, EBIT Margin)
def table19_calc(x, y):
	return np.where((x.isnull()) | (y.isnull()), np.nan, np.where((x == 0) & (y == 0), 0, np.where((x > 0) & (y == 0), 99999999, np.where((x < 0) & (y == 0), -99999999, np.where((x < 0) & (y < 0), -99999999, x / y )))))
 
## table21_calc:    Net Margin, EBITDA Margin, EBIT Margin
def table21_calc(x, y):
	return np.where((x.isnull()) | (y.isnull()), np.nan, np.where((x == 0) & (y == 0), 0, np.where((x > 0) & (y == 0), np.nan, np.where((x < 0) & (y < 0), -99999999, x / y))))


# debt_to_tnw_rto = tot_debt_amt / tangible_net_worth_amt			# table 19
# yrs_in_bus  =  final_form_date - bsd								# direct calc
# net_margin_rto =  net_inc_amt / tot_sales_amt						# table 19
 
## variables / ratio calculation 
 
hgc2012['cur_ast_amt'] = hgc2012.cur_rto * hgc2012.cur_liab_amt  
hgc2012['tot_debt_amt'] = hgc2012.debt_to_ebitda_rto * hgc2012.ebitda_amt   
hgc2012['net_worth_amt'] = hgc2012.tot_ast_amt - hgc2012.tot_liab_amt
hgc2012['yrs_in_bus'] = pd.Series(hgc2012.final_form_date - hgc2012.input_bsd.apply(lambda x: pd.to_datetime(x))) / np.timedelta64(1, 'Y')
hgc2012['net_margin_rto'] =  table19_calc(hgc2012.net_inc_amt, hgc2012.net_sales_amt)
hgc2012['debt_to_tnw_rto'] =  table19_calc(hgc2012.tot_debt_amt, hgc2012.tangible_net_worth_amt)
 
 
# read in the sic_indust data
sic_indust = pd.read_excel("H:\\work\\usgc\\2015\\quant\\SIC_Code_List.xlsx", sheetname = "sic_indust") 

hgc2012['indust'] = hgc2012.us_sic.replace(dict(zip(sic_indust.sic_code, sic_indust.indust)))
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
hgc2012_after_sic_wo_allmissing = hgc2012_after_sic_wo_allmissing.rename(columns = {'input_bsd': 'bsd', 'net_sales_amt': 'tot_sales_amt', 'input_dtnw': 'debt_to_nw_rto'})
hgc2012_after_sic_wo_allmissing['yeartype'] = '2012HBC'


common_vars = ['sk_entity_id', 'uen', 'final_form_date', 'us_sic', 'default_flag', 'default_date', 'yeartype', 'sector_group', 'quant_ranking', 'final_ranking']
final_model_vars = ['cur_ast_amt', 'tot_ast_amt', 'cur_liab_amt', 'tot_liab_amt', 'tot_debt_amt', 'net_worth_amt', 'tangible_net_worth_amt', 'tot_sales_amt', 'net_inc_amt', 'ebitda_amt', 'dsc', 'yrs_in_bus', 'debt_to_tnw_rto', 'debt_to_ebitda_rto', 'net_margin_rto', 'cur_rto']
hgc2012_after_sic_wo_allmissing.ix[:, common_vars + final_model_vars].to_excel("H:\\work\\usgc\\2015\\quant\\2015_supp\\F2012ALL_4_model.xlsx") 




