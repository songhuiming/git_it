
# 1: stratified sampling good to keep sample/population having same proportion on each sector
# 2: if debt_to_ebitda_rto < 0 or debt_to_tnw_rto < 0, then it should be replaced by the max(bad pd point since it's because of negative tnw and ebitda)
# 3: WoE transform for the samples and 2014 data
# 4: 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as spstat

model_factors = ['tot_sales_amt', 'dsc', 'yrs_in_bus', 'debt_to_tnw_rto', 'debt_to_ebitda_rto', 'net_margin_rto', 'cur_rto']

f2012 = pd.read_excel("H:\\work\\usgc\\2015\\quant\\2015_supp\\F2012ALL_4_model.xlsx") 			#{0: 909, 1: 41}
f2013_1 = pd.read_excel("H:\\work\\usgc\\2015\\quant\\2015_supp\\F2013HGC_4_model.xlsx") 		#{0: 1046, 1: 21}
f2013_2 = pd.read_excel("H:\\work\\usgc\\2015\\quant\\2015_supp\\F2013MI_4_model.xlsx") 		#{0: 991, 1: 12}
f2013 = pd.concat([f2013_1, f2013_2], axis = 0, ignore_index = True)
f2013 = f2013.query('tot_sales_amt >= 0') 														# ignore total sales < 0 obligors
f2014 = pd.read_excel("H:\\work\\usgc\\2015\\quant\\2015_supp\\F2014ALL_4_model.xlsx") 			#{0: 3572, 1: 41}
f2014 = f2014.query('tot_sales_amt >= 0')
devdata = pd.read_excel("H:\\work\\usgc\\2015\\quant\\2015_supp\\hgcdev_sic_remove_df.xlsx")

## negative debt_to_ebitda_rto and debt_to_tnw_rto should be replace by the max number
f2012['debt_to_ebitda_rto'] = np.where(f2012.debt_to_ebitda_rto < 0, f2012.debt_to_ebitda_rto.max(), f2012.debt_to_ebitda_rto)
f2012['debt_to_tnw_rto'] = np.where(f2012.debt_to_tnw_rto < 0, f2012.debt_to_tnw_rto.max(), f2012.debt_to_tnw_rto)
f2013['debt_to_ebitda_rto'] = np.where(f2013.debt_to_ebitda_rto < 0, f2013.debt_to_ebitda_rto.max(), f2013.debt_to_ebitda_rto)
f2013['debt_to_tnw_rto'] = np.where(f2013.debt_to_tnw_rto < 0, f2013.debt_to_tnw_rto.max(), f2013.debt_to_tnw_rto)
f2014['debt_to_ebitda_rto'] = np.where(f2014.debt_to_ebitda_rto < 0, f2014.debt_to_ebitda_rto.max(), f2014.debt_to_ebitda_rto)
f2014['debt_to_tnw_rto'] = np.where(f2014.debt_to_tnw_rto < 0, f2014.debt_to_tnw_rto.max(), f2014.debt_to_tnw_rto)
devdata['debt_to_ebitda_rto'] = np.where(devdata.debt_to_ebitda_rto < 0, devdata.debt_to_ebitda_rto.max(), devdata.debt_to_ebitda_rto)
devdata['debt_to_tnw_rto'] = np.where(devdata.debt_to_tnw_rto < 0, devdata.debt_to_tnw_rto.max(), devdata.debt_to_tnw_rto) 

### final columns needed
common_vars = ['sk_entity_id', 'uen', 'final_form_date', 'us_sic', 'default_flag', 'default_date', 'yeartype', 'sector_group', 'quant_ranking', 'final_ranking']
final_model_vars = ['cur_ast_amt', 'tot_ast_amt', 'cur_liab_amt', 'tot_liab_amt', 'tot_debt_amt', 'net_worth_amt', 'tangible_net_worth_amt', 'tot_sales_amt', 'net_inc_amt', 'ebitda_amt', 'dsc', 'yrs_in_bus', 'debt_to_tnw_rto', 'debt_to_ebitda_rto', 'net_margin_rto', 'cur_rto']


################################################################ numbers to be sampled out  ##################################################################
# development data read in(after remove AGRI/NONP)
hgcdev_good_num = pd.read_pickle("H:\\work\\usgc\\2015\\quant\\2015_supp\\hgcdev_good_sector").shape[0] + 0.0
hgcdev_bad_num = pd.read_pickle("H:\\work\\usgc\\2015\\quant\\2015_supp\\hgcdev_bad_sector").shape[0] + 0.0
hgcdev_rto = hgcdev_good_num / hgcdev_bad_num

# number of sampling goods on each segment for 2012  
good_num_2012 = np.floor(41 * hgcdev_rto * f2012.query('default_flag == 1').sector_group.value_counts() / [f2012.query('default_flag == 1').shape[0] + 0]) + 1      	#   
good_num_2013 = np.floor(33 * hgcdev_rto * f2013.query('default_flag == 1').sector_group.value_counts() / [f2013.query('default_flag == 1').shape[0] + 0]) + 1  		#
good_num_2014 = np.floor(41 * hgcdev_rto * f2014.query('default_flag == 1').sector_group.value_counts() / [f2014.query('default_flag == 1').shape[0] + 0]) + 1 			#


#sample for F2012

def stratified_sampling(pop_data, size_data):
	# sample goods from the good population data ( pop_data.ix[default_flag == 0] )
	# number of samples on each sector is given by size_data  ( good_num_2012 ) 
	sample_output_df = pd.DataFrame(columns = list(pop_data))
	good_grouped = pop_data.ix[pop_data.default_flag == 0].groupby('sector_group')
	bad_sector_groups = pop_data.ix[pop_data.default_flag == 0].sector_group.unique()
	init_rows = []
	for grp, grp_value in good_grouped:
		try:
			print "Good to be sampled for group %s is %s. " %(grp, size_data[grp])
			np.random.seed(9999)
			rows = np.random.choice(grp_value.index.values, size_data[grp], replace = False)
			init_rows.extend(rows)
			sample_df = pop_data.loc[rows]
			sample_output_df = pd.concat([sample_output_df, sample_df], axis = 0) 
			print sample_df.shape
			print sample_df.index
		except Exception:
			pass
	print init_rows
	return sample_output_df
	
f2012_good_samples = stratified_sampling(f2012, good_num_2012)

f2013_good_samples = stratified_sampling(f2013, good_num_2013)

f2014_good_samples = stratified_sampling(f2014, good_num_2014)

s2012 = pd.concat([f2012_good_samples, f2012.query('default_flag == 1')], axis = 0, ignore_index = True)
s2013 = pd.concat([f2013_good_samples, f2013.query('default_flag == 1')], axis = 0, ignore_index = True)
s2014 = pd.concat([f2014_good_samples, f2014.query('default_flag == 1')], axis = 0, ignore_index = True)

## final sample
final_sample = pd.concat([s2012, s2013, s2014, devdata], axis = 0, ignore_index = True)

##############################################################################  calculate WOE  #############################################################################

#1 DSC
dsc_cut = [0, 1, 1.25, 2.06452, 4.45]
dsc_woe = [-1.33433, -0.99743, -0.86373, 0.34456, 0.77557, 1.67578]

#2 cur_rto
cur_rto_cut = [0.87565, 1.27055, 1.89332]
cur_woe = [-1.13540, -0.18649, 0.32542, 1.30794]

#3 debt_to_tnw_rto
debt_2_tnw = [0, 0.61247, 1.28134, 2.48821, 4.55724] 
debt_2_tnw_woe = [-1.06007, 1.32242, 1.12546, 0.52609, 0.21725, -0.75127]

#4 years in business
yrs_in_b = [3, 10, 16.41667, 24.89589, 39.23836]
yrs_in_b_woe = [-1.11391, -0.46495, -0.24602, -0.03031, 0.61446, 1.04355]

#5 tot_sales_amt
tot_sales = [5000000, 20000000]
tot_sales_woe = [-0.56285, 0.46009, 1.22319]

#6 debt_to_ebitda_rto
dt_2_ebitda = [0, 2, 4.54977, 6.34321, 9.90462, 14.93103]
dt_2_ebitda_woe = [-1.27817, 1.40998, 1.00044, 0.43944, 0.20636, -0.604078, -0.82749]

# net_margin_rto
net_margin_cut = [0, 0.01572, 0.1321]
net_margin_woe = [-0.98827, 0.16849, 0.62193, 1.03296]

def woe(x, bin_x, woe_value, right=1):
	return [woe_value[i] for i in np.digitize(np.array(x), np.array(bin_x), right = right)]

###################     WoE 1: WoE transformation on the Sampled data 
#1 logic: dsc=na, _dsc=0; dsc=0, _dsc=-0.99743; the rest are right included; ebit<0, go to bad;
final_sample['_dsc'] = woe(final_sample.ix[:, u'dsc'], dsc_cut, dsc_woe)
final_sample[u'_dsc'][final_sample[u'dsc'].isnull()] = 0
final_sample[u'_dsc'][final_sample[u'dsc'] == 0] = -0.99743
final_sample[u'_dsc'][(final_sample[u'ebitda_amt'] < 0) & (final_sample[u'dsc'].isnull())] = -1.33433
final_sample[u'_dsc'].value_counts(dropna = 0).sort_index()
# dsc2 = [-inf, 0, 1, 1.25, 2.06452, 4.45, inf]
# pd.value_counts(pd.cut(final_sample[u'_dsc'], dsc2, right = 1), sort=1, dropna = 0)
 
# cur_rto:
final_sample[u'_cur_rto'] = woe(final_sample.ix[:, u'cur_rto'], cur_rto_cut, cur_woe)
final_sample[u'_cur_rto'][final_sample[u'cur_rto'].isnull()] = 0
final_sample[u'_cur_rto'][final_sample[u'cur_rto'] == 0.87565] = -1.13540
final_sample[u'_cur_rto'][(final_sample[u'cur_rto'].isnull()) & (final_sample[u'cur_liab_amt'] > 0) & (final_sample[u'cur_ast_amt'] <= 0)] = -1.13540 
final_sample[u'_cur_rto'].value_counts().sort_index()

#3: d_2_tnw=0: _d_2_tnw=1.32242; d_2_tnw=na & tnw<=0: _d_2_tnw=-1.06007  
final_sample[u'_debt_2_tnw'] = woe(final_sample.ix[:, u'debt_to_tnw_rto'], debt_2_tnw, debt_2_tnw_woe)
final_sample[u'_debt_2_tnw'][final_sample[u'debt_to_tnw_rto'].isnull()] = 0
final_sample[u'_debt_2_tnw'][final_sample[u'debt_to_tnw_rto'] == 0] = 1.32242
final_sample[u'_debt_2_tnw'][(final_sample[u'debt_to_tnw_rto'].isnull()) & (final_sample[u'tangible_net_worth_amt'] <=0)] = -1.06007 
final_sample[u'_debt_2_tnw'].value_counts().sort_index()

#4: yr_in_b = na = -1.11391, as sas code: if Yrs_In_B<0 then _Yrs_In_B=-1.11391;
final_sample[u'_yrs_in_b'] = woe(final_sample.ix[:, u'yrs_in_bus'], yrs_in_b, yrs_in_b_woe)
final_sample[u'_yrs_in_b'][final_sample[u'yrs_in_bus'].isnull()] = -1.11391
final_sample[u'_yrs_in_b'][final_sample[u'yrs_in_bus'] == 3] = -1.11391
final_sample[u'_yrs_in_b'][final_sample[u'yrs_in_bus'] < 0] = -1.11391
final_sample[u'_yrs_in_b'].value_counts().sort_index()

#5: left included in sas: low-<5000000   5000000-<20000000   20000000-High
final_sample[u'_tot_sales'] = woe(final_sample[u'tot_sales_amt'], tot_sales, tot_sales_woe, right = 0)
final_sample[u'_tot_sales'][final_sample[u'tot_sales_amt'].isnull()] = -0.56285;
final_sample[u'_tot_sales'].value_counts().sort_index()
# 2nd way to transform total sales: binned based on small busibess / business banking / mid-market threshold
final_sample[u'_tot_sales2'] = np.where(final_sample.tot_sales_amt > 20000000, 3, np.where(final_sample.tot_sales_amt > 5000000, 2, 1))

#6: 
final_sample[u'_dt_2_ebitda'] = woe(final_sample[u'debt_to_ebitda_rto'], dt_2_ebitda, dt_2_ebitda_woe)
final_sample[u'_dt_2_ebitda'][final_sample[u'debt_to_ebitda_rto'] == 0] = 1.40998
final_sample[u'_dt_2_ebitda'][final_sample[u'debt_to_ebitda_rto'].isnull()] = 0
final_sample[u'_dt_2_ebitda'][(final_sample[u'debt_to_ebitda_rto'].isnull()) & (final_sample[u'ebitda_amt'] <= 0)] = -1.27817
final_sample[u'_dt_2_ebitda'].value_counts().sort_index()

#7:
final_sample[u'_net_margin_rto'] = woe(final_sample[u'net_margin_rto'], net_margin_cut, net_margin_woe)
final_sample[u'_net_margin_rto'][final_sample[u'net_margin_rto'] == 0] = 0.16849
final_sample[u'_net_margin_rto'][final_sample[u'net_margin_rto'].isnull()] = 0
final_sample[u'_net_margin_rto'][(final_sample[u'net_margin_rto'].isnull()) & (final_sample[u'net_inc_amt'] < 0)] = -0.98827
final_sample[u'_net_margin_rto'].value_counts().sort_index()

## verify and compare the result with SAS
model_var_woe = [x for x in list(final_sample) if x.startswith('_')]
for i in model_var_woe:
	print "This is for factor : %s" %(i)
	print 'Counts on each WOE value' + '-' * 30
	print final_sample.ix[:, i].value_counts().sort_index()
	print 'Counts on each WOE value' + '-' * 30
	print final_sample.ix[:, i].value_counts().sort_index() / final_sample.shape[0]
	print 'Default rate on each value of WOE' + '-' * 30
	print final_sample.groupby(i).default_flag.mean()    			# not all in ascending order #
	
###################     WoE 2: WoE transformation on the F2014 Data


####################################################################
def auc(x, y):
	unq_x = np.unique(x)
	n1 = sum(y)
	n = len(y)
	Sens = np.zeros_like(unq_x)
	Spec = np.zeros_like(unq_x)
	for j, u in enumerate(unq_x):
		Sens[j] = np.sum((x >= u) * y) / float(n1)
		Spec[j] = np.sum((x <= u) *(1 - y)) / float(n - n1)
	auc = 0.0
	for i in range(len(Spec) - 1):
		auc += (Spec[i + 1] - Spec[i]) * (Sens[i + 1] + Sens[i]) / 2.0
	return auc
	

def avg_pd_on_bin(varx, vary, n = 10):
	'''
	the purpose is to check the relationship between independent variable varx and dependent variable vary. 
	First need to impute the missing values by the worse end(higher pd end) 25 percentile
	then cut varx into n = 10 bins, with similar number of obs in each bin (cut by percentiles)
	calculate average pd in each interval
	calculate AUC or gini coefficient
	'''
	var_x = varx.name
	try:
		if var_x in ['debt_to_ebitda_rto', 'debt_to_tnw_rto']:
			newx = varx.fillna(np.percentile(varx[(varx != -99999999) & (varx != 99999999)], 75))
		else:
			newx = varx.fillna(np.percentile(varx[(varx != -99999999) & (varx != 99999999)], 25))
		bins = pd.Series(pd.qcut(newx.values, n))
		df = pd.concat([newx, vary, bins], axis = 1, ignore_index = True)
		df.columns = ['newx', 'vary', 'bins']
		grpby_result = df.groupby('bins')
	except Exception:
		pass
	grpby_x = grpby_result.newx.agg([np.min, np.max, np.mean])
	grpby_total = grpby_result.size()
	grpby_total_bad =  grpby_result.vary.sum()
	grpby_total_good = grpby_total - grpby_total_bad 
	grpby_avg_pd = grpby_result.vary.mean()
	grpby_summary = pd.concat([grpby_x, grpby_total, grpby_total_bad,  grpby_avg_pd], axis = 1, ignore_index = True)
	grpby_summary.columns = ['min_x', 'max_x', 'avg_x', 'total_counts', 'total_bad_counts',  'average_PD']
	return grpby_summary.ix[:, ['avg_x', 'total_counts', 'total_bad_counts',  'average_PD']].sort('avg_x') 
	#return grpby_summary.sort('min_x')  #final_bin.sort('vary', ascending = False)	

"""	 Temp comment out this part 
# Analysis 1: Gini Coeffifient + average PD on each variable 
for i in model_factors:
	print '=' * 20 + ' ' + i + ' ' + '=' * 20
	if i in ['debt_to_ebitda_rto', 'debt_to_tnw_rto']:
		gini_coeff = 2 * auc(final_sample.ix[:, i], final_sample.default_flag) - 1
	else:
		gini_coeff = 2 * auc(-final_sample.ix[:, i], final_sample.default_flag) - 1
	print 'Gini Coeffifient is: %s' % gini_coeff
	avg_pd = avg_pd_on_bin(final_sample.ix[:, i], final_sample.default_flag)
	print avg_pd.to_string()

# Analysis 2: calculate the drivers median on each FRR	
for i in model_factors:
	print '=' * 20 + ' ' + i + ' ' + '=' * 20
	print final_sample.ix[:, ['final_ranking', i]].groupby('final_ranking').median().reset_index()
"""

###########################################################    WoE Calculation Function    ############################################
# woe = log( (ngood/tgood +1e-10) / (nbad/tbad +1e-10) ) ;
def woe_calc(varx, vary, n = 30, missing_method = 2):
	'''
	calculate WoE: split data to n bins equally based on the percentiles
	check if the average PD is monotonic on the split above: if not, split again to n - 1. repeat until monotonic
	if X has missing, impute it by percentiles or mean or treat missing as an bin itself
	'''
	r = 0
	while r < 1:
		try:
			if missing_method == 1: 				#method 1: fill NA by percentiles
				newx = varx.fillna(np.percentile(varx[(varx != -99999999) & (varx != 99999999)], 25))
				bins = pd.Series(pd.qcut(newx.values, n))
				df = pd.concat([newx, vary, bins], axis = 1, ignore_index = True)
				df.columns = ['newx', 'vary', 'bins']
				grpby_result = df.groupby('bins')
				r = abs(spstat.spearmanr(grpby_result.newx.mean(), grpby_result.vary.mean())[0])
			elif missing_method == 2:				#method 2: fill NA by mean
				newx = varx.fillna(varx[(varx != -99999999) & (varx != 99999999)].mean())
				bins = pd.Series(pd.qcut(newx.values, n))
				df = pd.concat([newx, vary, bins], axis = 1, ignore_index = True)
				df.columns = ['newx', 'vary', 'bins']
				grpby_result = df.groupby('bins')
				r = abs(spstat.spearmanr(grpby_result.newx.mean(), grpby_result.vary.mean())[0])
			elif missing_method == 3:
				bins = pd.Series(pd.qcut(varx.values, n))   		# .fillna(-np.inf)
				df = pd.concat([varx, vary, bins], axis = 1, ignore_index = True)
				df.columns = ['newx', 'vary', 'bins']
				grpby_result = df.groupby('bins')
				r = abs(spstat.spearmanr(grpby_result.newx.mean(), grpby_result.vary.mean())[0])
			# print grpby_result.newx.mean()
			# print grpby_result.vary.mean()
		except Exception:
			pass
		n -= 1
	grpby_min = grpby_result.newx.min()
	grpby_max = grpby_result.newx.max()
	grpby_total = grpby_result.size()
	grpby_total_bad =  grpby_result.vary.sum()
	grpby_total_good = grpby_total - grpby_total_bad 
	#grpby_woe = np.log( (grpby_total_good / (grpby_total_good.sum()) + 1e-5) / (grpby_total_bad / grpby_total_bad.sum() + + 1e-5))
	grpby_summary = pd.concat([grpby_min, grpby_max, grpby_total, grpby_total_bad, grpby_total_good], axis = 1, ignore_index = True)
	grpby_summary.columns = ['min_x', 'max_x', 'total_counts', 'total_bad_counts', 'total_good_counts']
	final_bin = pd.concat([grpby_result.newx.mean(), grpby_result.vary.mean()], axis = 1).reset_index().sort('newx')
	if missing_method == 3:
		#missing_value = pd.DataFrame({'bins': 'Missing', 'newx': np.nan, 'vary': vary[varx.isnull()].mean()}, index = [0]) 
		#final_bin = pd.concat([final_bin, missing_value], axis = 0, ignore_index = True)
		missing_value = pd.DataFrame({'min_x': np.nan, 'max_x': np.nan, 'total_counts': varx.isnull().sum(), 'total_bad_counts': vary[varx.isnull()].sum(), 'total_good_counts': varx.isnull().sum() - vary[varx.isnull()].sum()}, index = [0]) 
		grpby_summary = pd.concat([grpby_summary, missing_value], axis = 0, ignore_index = True)
	grpby_summary['WoE'] = np.log( (grpby_summary.total_good_counts / (grpby_summary.total_good_counts.sum()) + 1e-5) / (grpby_summary.total_bad_counts / grpby_summary.total_bad_counts.sum() + + 1e-5) )
	grpby_summary['Average_PD'] = grpby_summary.total_bad_counts / grpby_summary.total_counts
	return grpby_summary.sort('WoE')  #final_bin.sort('vary', ascending = False)	






