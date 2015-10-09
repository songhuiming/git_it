
# 1: stratified sampling good to keep sample/population having same proportion on each sector
# 2: if debt_to_ebitda_rto < 0 or debt_to_tnw_rto < 0, then it should be replaced by the max(bad pd point since it's because of negative tnw and ebitda)
# 3: WoE transform for the samples and 2014 data
 
# pd.set_option('display.height', 1000)
# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats 

model_factors = ['tot_sales_amt', 'dsc', 'yrs_in_bus', 'debt_to_tnw_rto', 'debt_to_ebitda_rto', 'net_margin_rto', 'cur_rto']

f2012 = pd.read_excel("H:\\work\\usgc\\2015\\quant\\2015_supp\\F2012ALL_4_model.xlsx") 			#{0: 909, 1: 41}
f2013_1 = pd.read_excel("H:\\work\\usgc\\2015\\quant\\2015_supp\\F2013HGC_4_model.xlsx") 		#{0: 1046, 1: 21}
f2013_2 = pd.read_excel("H:\\work\\usgc\\2015\\quant\\2015_supp\\F2013MI_4_model.xlsx") 		#{0: 991, 1: 12}
f2013 = pd.concat([f2013_1, f2013_2], axis = 0, ignore_index = True)
f2013 = f2013.query('tot_sales_amt >= 0 & final_ranking > 0') 									# ignore total sales < 0 and no default ranking obligors
f2014 = pd.read_excel("H:\\work\\usgc\\2015\\quant\\2015_supp\\F2014ALL_4_model.xlsx") 			#{0: 3572, 1: 41}
f2014 = f2014.query('tot_sales_amt >= 0 & final_ranking > 0')									# no default ranking data	
devdata = pd.read_excel("H:\\work\\usgc\\2015\\quant\\2015_supp\\hgcdev_sic_remove_df.xlsx")

## negative debt_to_ebitda_rto and debt_to_tnw_rto should be replace by the max number
# f2012['debt_to_ebitda_rto'] = np.where(f2012.debt_to_ebitda_rto < 0, f2012.debt_to_ebitda_rto.max(), f2012.debt_to_ebitda_rto)
# f2012['debt_to_tnw_rto'] = np.where(f2012.debt_to_tnw_rto < 0, f2012.debt_to_tnw_rto.max(), f2012.debt_to_tnw_rto)
# f2013['debt_to_ebitda_rto'] = np.where(f2013.debt_to_ebitda_rto < 0, f2013.debt_to_ebitda_rto.max(), f2013.debt_to_ebitda_rto)
# f2013['debt_to_tnw_rto'] = np.where(f2013.debt_to_tnw_rto < 0, f2013.debt_to_tnw_rto.max(), f2013.debt_to_tnw_rto)
# f2014['debt_to_ebitda_rto'] = np.where(f2014.debt_to_ebitda_rto < 0, f2014.debt_to_ebitda_rto.max(), f2014.debt_to_ebitda_rto)
# f2014['debt_to_tnw_rto'] = np.where(f2014.debt_to_tnw_rto < 0, f2014.debt_to_tnw_rto.max(), f2014.debt_to_tnw_rto)
# devdata['debt_to_ebitda_rto'] = np.where(devdata.debt_to_ebitda_rto < 0, devdata.debt_to_ebitda_rto.max(), devdata.debt_to_ebitda_rto)
# devdata['debt_to_tnw_rto'] = np.where(devdata.debt_to_tnw_rto < 0, devdata.debt_to_tnw_rto.max(), devdata.debt_to_tnw_rto) 

f121314 = pd.concat([f2012, f2013, f2014], axis = 0, ignore_index = True)

### final columns needed
common_vars = ['sk_entity_id', 'uen', 'final_form_date', 'us_sic', 'default_flag', 'default_date', 'yeartype', 'sector_group', 'quant_ranking', 'final_ranking']
final_model_vars = ['cur_ast_amt', 'tot_ast_amt', 'cur_liab_amt', 'tot_liab_amt', 'tot_debt_amt', 'net_worth_amt', 'tangible_net_worth_amt', 'tot_sales_amt', 'net_inc_amt', 'ebitda_amt', 'dsc', 'yrs_in_bus', 'debt_to_tnw_rto', 'debt_to_ebitda_rto', 'net_margin_rto', 'cur_rto']


##################################################################   01: Stratified Sampling    ##################################################################
# development data read in(after remove AGRI/NONP)
hgcdev_good_num = pd.read_pickle("H:\\work\\usgc\\2015\\quant\\2015_supp\\hgcdev_good_sector").shape[0] + 0.0
hgcdev_bad_num = pd.read_pickle("H:\\work\\usgc\\2015\\quant\\2015_supp\\hgcdev_bad_sector").shape[0] + 0.0
hgcdev_rto = hgcdev_good_num / hgcdev_bad_num

# number of sampling goods on each segment for 2012  
#good_num_2012 = np.floor(41 * hgcdev_rto * f2012.query('default_flag == 1').sector_group.value_counts() / [f2012.query('default_flag == 1').shape[0] + 0]) + 1      	#   
#good_num_2013 = np.floor(33 * hgcdev_rto * f2013.query('default_flag == 1').sector_group.value_counts() / [f2013.query('default_flag == 1').shape[0] + 0]) + 1  		#
#good_num_2014 = np.floor(41 * hgcdev_rto * f2014.query('default_flag == 1').sector_group.value_counts() / [f2014.query('default_flag == 1').shape[0] + 0]) + 1 			#
good_num_2012 = np.ceil(f2012.sector_group.value_counts(dropna = False).sort_index() / f2012.shape[0] * 41 * hgcdev_rto)
good_num_2013 = np.ceil(f2013.sector_group.value_counts(dropna = False).sort_index() / f2013.shape[0] * 33 * hgcdev_rto)
good_num_2014 = np.ceil(f2014.sector_group.value_counts(dropna = False).sort_index() / f2014.shape[0] * 41 * hgcdev_rto)


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
final_sample['year2'] = [x[:4] for x in final_sample.yeartype]

final_sample.to_excel('H:\\work\\usgc\\2015\\quant\\2015_supp\\final_sample.xlsx')
full_data = pd.concat([devdata, f2012, f2013, f2014], axis = 0, ignore_index = True)
full_data.to_excel('H:\\work\\usgc\\2015\\quant\\2015_supp\\full_data.xlsx')


#########################################################   02: Some Functions for Future Use  ####################################################
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
	

def avg_pd_on_bin(varx, vary, n = 7):
	'''
	the purpose is to check the relationship between independent variable varx and dependent variable vary. 
	First need to impute the missing values by the worse end(higher pd end) 25 percentile
	then cut varx into n = 10 bins, with similar number of obs in each bin (cut by percentiles)
	calculate average pd in each interval
	calculate AUC or gini coefficient
	'''
	var_x = varx.name
	try:
		newx = varx
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
# frr_x_median = pd.DataFrame(columns = model_factors)
# for i in model_factors:
	# print '=' * 20 + ' ' + i + ' ' + '=' * 20
	# res = final_sample.ix[:, ['final_ranking', i]].groupby('final_ranking').median().reset_index().ix[:, 1]
	# print res
	# frr_x_median[i] = res
	# print frr_x_median.to_string()
"""
## drivers median on FRR
print '='*50 +' Final Sample Drivers Median on FRR ' + '='*50
print final_sample.ix[:, model_factors + ['final_ranking']].groupby('final_ranking').median().to_string()

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
			if missing_method == 1: 				#method 1: fill NA 
				newx = varx 
				bins = pd.Series(pd.qcut(newx.values, n))
				df = pd.concat([newx, vary, bins], axis = 1, ignore_index = True)
				df.columns = ['newx', 'vary', 'bins']
				grpby_result = df.groupby('bins')
				r = abs(stats.spearmanr(grpby_result.newx.mean(), grpby_result.vary.mean())[0])
			elif missing_method == 2:  									#method 2: treat missing as a level
				bins = pd.Series(pd.qcut(varx.values, n))   		# .fillna(-np.inf)
				df = pd.concat([varx, vary, bins], axis = 1, ignore_index = True)
				df.columns = ['newx', 'vary', 'bins']
				grpby_result = df.groupby('bins')
				r = abs(stats.spearmanr(grpby_result.newx.mean(), grpby_result.vary.mean())[0])
			else:
				print "Please select Missing value processing method"
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

from sklearn import metrics

def gini2(x, y):
	fpr, tpr, thresholds = metrics.roc_curve(y, x)
	return 2 * float(metrics.auc(fpr, tpr)) - 1




