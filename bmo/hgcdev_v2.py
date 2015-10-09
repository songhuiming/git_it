
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import bisect

hgcdev_all = pd.read_excel(u"H:\\work\\usgc\\2015\\quant\\hgcdev0311g_shm.xlsx", sheetname = "data")
varhgcdev = [x.lower() for x in list(hgcdev_all)]
hgcdev_all.columns = varhgcdev

hgcdev = hgcdev_all.query('seg2 > 1')

dev_vars = ['sk_entity_id', 'sic_code', 'sector_group', 'default_date', 'default_flag', 'current_assets', 'current_liabilities', 'current_ratio', 'total_assets', 'total_debt', 'ebitda',
'total_debt_service_amt', 'total_operating_income', 'total_sales', 'debt_to_ebitda', 'debt_to_tangible_net_worth', 'debt_service_coverage', 'net_margin', 'tnw', 'net_profit', 'yrs_in_business']

dev_rename = {'sic_code': 'us_sic', 'current_assets': 'cur_ast_amt', 'current_liabilities': 'cur_liab_amt', 'current_ratio': 'cur_rto', 'total_assets': 'tot_ast_amt', 
'total_debt': 'tot_debt_amt', 'total_sales': 'tot_sales_amt', 'debt_to_ebitda': 'debt_to_ebitda_rto', 'debt_to_tangible_net_worth': 'debt_to_tnw_rto', 'ebitda': 'ebitda_amt',
'debt_service_coverage': 'dsc', 'net_margin': 'net_margin_rto', 'tnw': 'tangible_net_worth_amt', 'net_profit': 'net_inc_amt', 'yrs_in_business': 'yrs_in_bus'}

hgcdev['default_flag'] = np.where(hgcdev.default_flag == 'Y', 1, np.where(hgcdev.default_flag == 'N', 0, np.nan))

# pick the non-missing default_flag and remove agri non-profit obligors
hgcdev_sic = hgcdev.query('default_flag >= 0  & sector_group != "AGRI" & sector_group != "NONP" ')
hgcdev_sic_final = hgcdev_sic.ix[:, dev_vars]
hgcdev_sic_final['yeartype'] = '2011Before'
hgcdev_sic_final = hgcdev_sic_final.rename(columns = dev_rename)
hgcdev_sic_final.to_excel("H:\\work\\usgc\\2015\\quant\\2015_supp\\hgcdev_sic_remove_df.xlsx")

# the following is not good, because it includes Non-Profit / Agri / Govn
# good ones: proportion by sector_group
# hgcdev.query('default_flag == "N"').sector_group.value_counts() / len(hgcdev.query('default_flag == "N"').sector_group)
# hgcdev.query('default_flag == "Y"').sector_group.value_counts() / len(hgcdev.query('default_flag == "Y"').sector_group)
# sum(hgcdev.default_flag == 'Y')  		# 174
# sum(hgcdev.default_flag == 'N')		# 898

#################################  prepare calc for stratified sampling, excluding AGRI and NON-Profit  ##############################
# overall proportion by sector_group
hgcdev.sector_group.value_counts() / len(hgcdev.sector_group)

hgcdev_good_sector = hgcdev_sic.query('default_flag == 0')
hgcdev_bad_sector = hgcdev_sic.query('default_flag == 1')

print hgcdev_good_sector.shape        # 804 Good
print hgcdev_bad_sector.shape			# 168 Bad

hgcdev_good_sector.sector_group.value_counts() 
# SRVC    245
# MFG     148
# WHLS    121
# CONS     88
# RETL     85
# REAL     65
# TRAN     32
# FIN      15
# GOVT      3
# MINE      2

# F2012 has 58 defaults, so need 58 * 804 / 168 = 278 goods  (after SIC, there is only 45 defaults, so need 45 * 804 / 168 = 220 goods)  
# F2013 has 45 defaults, so need 45 * 804 / 168 = 216 goods
# F2014 has 49 defaults, so need 49 * 804 / 168 = 235 goods

hgvdev_prop = hgcdev_good_sector.groupby('sector_group').size() / hgcdev_good_sector.shape[0]
# CONS            0.109453
# FIN             0.018657
# GOVT            0.003731
# MFG             0.184080
# MINE            0.002488
# REAL            0.080846
# RETL            0.105721
# SRVC            0.304726
# TRAN            0.039801
# WHLS            0.150498

# number of sampling goods on each segment for 2012  
good_num_2012 = np.floor(43 * hgcdev_good_sector.sector_group.value_counts() / len(hgcdev_bad_sector.sector_group)) + 1      	#  209 goods to be sampled, compared with 43 bads
good_num_2013 = np.floor(49 * hgcdev_good_sector.sector_group.value_counts() / len(hgcdev_bad_sector.sector_group)) + 1 		#
good_num_2014 = np.floor(42 * hgcdev_good_sector.sector_group.value_counts() / len(hgcdev_bad_sector.sector_group)) + 1			#


#sample for F2012

def stratified_sampling(pop_data, size_data):
	# sample goods from the good population data ( pop_data.ix[default_flag == 0] )
	# number of samples on each sector is given by size_data  ( good_num_2012 ) 
	sample_output_df = pd.DataFrame(columns = list(pop_data))
	good_grouped = pop_data.ix[pop_data.default_flag == 0].groupby('sector_group')
	for grp, grp_value in good_grouped:
		print "Good to be sampled for group %s is %s. " %(grp, size_data[grp])
		np.random.seed(9999)
		rows = np.random.choice(grp_value.index.values, size_data[grp])
		sample_df = pop_data.ix[rows]
		sample_output_df = pd.concat([sample_output_df, sample_df], axis = 0) 
		print sample_df.shape
		print sample_df.index
	return sample_output_df
	
f2012_good_samples = stratified_sampling(hgc2012_final, good_num_2012)

 












