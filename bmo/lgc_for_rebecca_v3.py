# prep good / bad separately 

import pandas as pd
import numpy as np
from datetime import datetime
import statsmodels.formula.api as smf
import scipy.stats as stats

lcmfile = pd.read_csv(r'R:\Global_Market_Risk\RCST\WCM\Huiming_Song\LCM\lcm_data.csv')
deffile = pd.read_csv(r'R:\Global_Market_Risk\RCST\WCM\Rebecca Sun\LCM\Data for Huiming\default_list.csv')
comp = pd.read_csv(r'R:\Global_Market_Risk\RCST\WCM\Rebecca Sun\LCM\Data for Huiming\compare PW\PW2010_CLEAN.csv')
reb_def = pd.read_excel('R:\Global_Market_Risk\RCST\WCM\Huiming_Song\LCM\defaulters_rebecca_list.xlsx')

lcmfile.columns = [x.lower() for x in lcmfile.columns]
deffile.columns = [x.lower() for x in deffile.columns]
 
# remove duplicated defaults
deffile = deffile.sort(['entityuen', 'default_date']).drop_duplicates('entityuen')

# exclude data with excl = 1
lcmfile = lcmfile.query('excl == 0')

# change string to datetime
def str2time(x):
	return datetime.strptime(x, '%m/%d/%Y')

lcmfile.final_form_date = map(str2time, lcmfile.final_form_date)
lcmfile.financial_statement_date = map(str2time, lcmfile.financial_statement_date)
deffile.default_date = map(str2time, deffile.default_date)

# get the final_form_date year month
lcmfile['year'] = map(lambda x: x.year, lcmfile.final_form_date)
lcmfile['month'] = map(lambda x: x.month, lcmfile.final_form_date)
# get the performance window year for goods 
lcmfile['pw_year'] = np.where(lcmfile.month > 1, lcmfile.year, lcmfile.year - 1)

# get default year by pw
deffile['default_year'] = map(lambda x: x.year, deffile.default_date)
deffile['default_month'] = map(lambda x: x.month, deffile.default_date)
deffile['pw_default_year'] = np.where(deffile.default_month <= 10, deffile.default_year, deffile.default_year + 1)

# split data to good / bad
lcmgood = lcmfile[~(lcmfile.entityuen.isin(deffile.entityuen))]
lcmbad = lcmfile[(lcmfile.entityuen.isin(deffile.entityuen))]


def lcm_good_dataprep(pwYear):
	lcm_yr_data = lcmfile.ix[lcmfile.pw_year == pwYear, :]
	def_data = deffile.ix[deffile.pw_default_year <= pwYear + 1, :]
	lcm_yr_data = pd.merge(lcm_yr_data, def_data, on = 'entityuen', how = 'left')
	lcm_yr_data['default_flag'] = np.where(lcm_yr_data.default_date > 0, 1, 0)
	# pick only goods
	lcm_yr_data = lcm_yr_data.query('default_flag == 0')

	################################################################ calculate time interval, Performance Window   ######################################
	 
	lcm_yr_data['fyBegin'] = datetime.strptime(r'11/1/'+str(pwYear), '%m/%d/%Y')
	# delete all default obligors that default date is before the fiscal year start date
	lcm_yr_data = lcm_yr_data.ix[~((lcm_yr_data.default_date < lcm_yr_data.fyBegin) & (lcm_yr_data.default_date > 0)), :]
	
	lcm_yr_data['dura_fybegin_attst'] = (lcm_yr_data.ix[:, u'final_form_date'] - lcm_yr_data.ix[:, u'fyBegin']) / np.timedelta64(1, 'M')
	lcm_yr_data['dura_fs_attst'] = (lcm_yr_data.ix[:, u'final_form_date'] - lcm_yr_data.ix[:, u'financial_statement_date']) / np.timedelta64(1, 'M')
	lcm_yr_data['dura_fybegin_attst_abs'] = abs(lcm_yr_data.ix[:, 'dura_fybegin_attst'])

	lcm_yr_data.ix[:, 'pw'] = 0
	lcm_yr_data.ix[((-9 <= lcm_yr_data['dura_fybegin_attst']) & (lcm_yr_data['dura_fybegin_attst'] <0)), 'pw'] = 1     
	lcm_yr_data.ix[((0 <= lcm_yr_data['dura_fybegin_attst']) & (lcm_yr_data['dura_fybegin_attst'] <= 3)), 'pw'] = 2
	lcm_yr_data.ix[(lcm_yr_data['dura_fs_attst'] > 15), 'pw'] = 9

	lcm_yr_data['mkey'] = lcm_yr_data.ix[:, u'entityuen'].map(str) + lcm_yr_data.ix[:, u'pw_year'].map(str)
		
	lcm_yr_data_sort = lcm_yr_data.sort(['mkey', 'default_flag', 'pw', 'dura_fybegin_attst_abs'], ascending = [True, False, True, True])
	lcm_yr_data_after_pw = lcm_yr_data_sort.ix[((lcm_yr_data_sort['pw'] != 0) & (lcm_yr_data_sort['pw'] != 9)), :].drop_duplicates(['mkey']) 
	lcm_yr_data_after_pw['pw_year'] = lcm_yr_data_after_pw['pw_year'] + 1
	return lcm_yr_data_after_pw



def lcm_bad_dataprep(pwYear):
	lcm_yr_data = pd.merge(deffile, lcmfile, on = 'entityuen', how = 'left')
	lcm_yr_data['default_flag'] = np.where(lcm_yr_data.default_date > 0, 1, 0)
	
	lcm_yr_data['fyBegin'] = datetime.strptime(r'11/1/'+str(pwYear), '%m/%d/%Y')
	lcm_yr_data['fyEnd'] = datetime.strptime(r'10/31/'+str(pwYear + 1), '%m/%d/%Y')
	
	lcm_yr_data = lcm_yr_data.query('final_form_date < default_date &  fyBegin <= default_date <= fyEnd')
	
	lcm_yr_data['dura_fybegin_attst'] = (lcm_yr_data.ix[:, u'final_form_date'] - lcm_yr_data.ix[:, u'fyBegin']) / np.timedelta64(1, 'M')
	lcm_yr_data['dura_fybegin_attst_abs'] = abs(lcm_yr_data.ix[:, 'dura_fybegin_attst'])
	lcm_yr_data['dura_attst_df'] = (lcm_yr_data.ix[:, u'default_date'] - lcm_yr_data.ix[:, u'final_form_date']) / np.timedelta64(1, 'M')
	lcm_yr_data['dura_fs_df'] = (lcm_yr_data.ix[:, u'default_date'] - lcm_yr_data.ix[:, u'financial_statement_date']) / np.timedelta64(1, 'M')

	lcm_yr_data.ix[:, 'pw'] = 0
	
	lcm_yr_data.ix[(-12 <= lcm_yr_data['dura_fybegin_attst']) & (lcm_yr_data['dura_fybegin_attst'] <= 0) & ((3 <= lcm_yr_data['dura_attst_df']) & (lcm_yr_data['dura_attst_df'] <= 12)), 'pw'] = 1
	lcm_yr_data.ix[(-12 <= lcm_yr_data['dura_fybegin_attst']) & (lcm_yr_data['dura_fybegin_attst'] <= 0) & ((12 < lcm_yr_data['dura_attst_df']) & (lcm_yr_data['dura_attst_df'] <= 18)), 'pw'] = 2
	lcm_yr_data.ix[(-12 <= lcm_yr_data['dura_fybegin_attst']) & (lcm_yr_data['dura_fybegin_attst'] <= 0) & ((0 <= lcm_yr_data['dura_attst_df']) & (lcm_yr_data['dura_attst_df'] < 3)), 'pw'] = 3
	lcm_yr_data.ix[(-12 <= lcm_yr_data['dura_fybegin_attst']) & (lcm_yr_data['dura_fybegin_attst'] <= 0) & ((0 > lcm_yr_data['dura_attst_df']) | (lcm_yr_data['dura_attst_df'] > 18)), 'pw'] = 5
	
	lcm_yr_data.ix[((0 <= lcm_yr_data['dura_attst_df']) & (lcm_yr_data['dura_fybegin_attst'] >= 0)), 'pw'] = 4
	lcm_yr_data.ix[(lcm_yr_data['dura_fybegin_attst'] < -12), 'pw'] = 5
	
	lcm_yr_data.ix[((lcm_yr_data['dura_attst_df'] <= 0) | (lcm_yr_data['dura_fs_df'] > 24)), 'pw'] = 9

	lcm_yr_data['mkey'] = lcm_yr_data.ix[:, u'entityuen'].map(str) + lcm_yr_data.ix[:, u'pw_year'].map(str)
		
	lcm_yr_data_sort = lcm_yr_data.sort(['entityuen', 'pw', 'dura_fybegin_attst_abs'], ascending = [True, True, True])
	lcm_yr_data_after_pw = lcm_yr_data_sort.ix[((lcm_yr_data_sort['pw'] != 0) & (lcm_yr_data_sort['pw'] != 9)), :].drop_duplicates(['entityuen']) 
	lcm_yr_data_after_pw['pw_year'] = lcm_yr_data_after_pw['pw_default_year'] 
	return lcm_yr_data_after_pw


final_default_data = pd.DataFrame()
for i in range(2008, 2016):
	try:
		print i
		output = lcm_bad_dataprep(i)
		print output.shape
		final_default_data = pd.concat([final_default_data, output], axis = 0)
	except ValueError:
		pass

final_good_data = pd.DataFrame()
for i in range(2008, 2016):
	try:
		print i
		output = lcm_good_dataprep(i)
		print output.shape
		final_good_data = pd.concat([final_good_data, output], axis = 0)
	except ValueError:
		pass		

# concat bad good together, good only to 2014		
data_after_pw = pd.concat([final_default_data, final_good_data.query('pw_year <= 2014')], axis = 0)
 
## table19_calc:  Non-Debt Based Ratios ( except Net Margin, EBITDA Margin, EBIT Margin)
def table19_calc(x, y):
	return np.where((x.isnull()) | (y.isnull()), np.nan, np.where((x == 0) & (y == 0), 0, np.where((x > 0) & (y == 0), 100000, np.where((x < 0) & (y == 0), -100000, np.where((x < 0) & (y < 0), -100000, x / y )))))

## table20_calc:   Debt Based Ratios
def table20_calc(x, y):
	return np.where((x.isnull()) | (y.isnull()), np.nan, np.where((x == 0) & (y == 0), 0, np.where((x > 0) & (y == 0), np.nan, np.where((x > 0) & (y < 0), 100000, x / y))))
	
## table21_calc:    Net Margin, EBITDA Margin, EBIT Margin
def table21_calc(x, y):
	return np.where((x.isnull()) | (y.isnull()), np.nan, np.where((x == 0) & (y == 0), 0, np.where((x > 0) & (y == 0), np.nan, np.where((x < 0) & (y < 0), -100000, x / y))))

# years in business 	
def yib_calc(x, y):
	return (x - y) / np.timedelta64(1, 'Y')


data_after_pw['interest_expense'] = map(lambda x: 0 if x == '.' else float(x), data_after_pw.interest_expense)
data_after_pw['obs_imputed_interest'] = map(lambda x: 0 if x == '.' else float(x), data_after_pw.obs_imputed_interest)
data_after_pw['interest'] = data_after_pw['interest_expense'] + data_after_pw['obs_imputed_interest'] 

data_after_pw['debt'] = map(lambda x: 0 if x == '.' else float(x), data_after_pw.debt)
data_after_pw['bookequity'] = map(lambda x: 0 if x == '.' else float(x), data_after_pw.bookequity)
data_after_pw['book_cap'] = data_after_pw['debt'] + data_after_pw['bookequity'] 

data_after_pw['ebitda'] = map(lambda x: 0 if x == '.' else float(x), data_after_pw.ebitda)
data_after_pw['ni_excl_extraord_items'] = map(lambda x: 0 if x == '.' else float(x), data_after_pw.ni_excl_extraord_items)
data_after_pw['total_sales'] = map(lambda x: 0 if x == '.' else float(x), data_after_pw.total_sales)
 
raw_var = ['ebitda', 'interest', 'ni_excl_extraord_items', 'total_sales', 'debt', 'book_cap']
 
data_after_pw['new_ebitda_interest'] = table19_calc(data_after_pw.ebitda, data_after_pw.interest) 	
data_after_pw['new_net_margin'] = table21_calc(data_after_pw.ni_excl_extraord_items, data_after_pw.total_sales) 	
data_after_pw['new_debt_bookcap'] = table20_calc(data_after_pw.debt, data_after_pw.book_cap) 	
data_after_pw['new_debt_ebitda'] = table20_calc(data_after_pw.debt, data_after_pw.ebitda) 	

# remove obs with unreasonable data
data_after_pw = data_after_pw.query('interest >=0 & total_sales > 0 & book_cap > 0 & debt >= 0')
# adjust the negative values (new_debt_bookcap new_debt_ebitda are all positive, don't need to do it now)
# select only some columns
data_after_pw = data_after_pw.ix[:, ['sic_sector', 'pw_year', 'new_ebitda_interest', 'new_net_margin', 'new_debt_bookcap', 'new_debt_ebitda', 'frr_rank', 'default_flag']]


def normalize_train(indata = data_after_pw, min_c = 5, max_c = 95):
	'''
	1: assign -99999999 and 99999999 as missing  
	2: calculate 5/95 percentile from non-missing data
	3: floor / cap the data  (dev + 12 13 14 sample, )
	4: impute missing by the worst side 3rd quantile (or median)
	5: replace -99999999 or 99999999 by min/max, that is, the 5/95 percentile above 
	6: output: 'output' is data summary info('floor_value', 'cap_value', 'impute_value', 'mean_value', 'std_value') for transform, and the values to use on test data
	7: the new variable ended with _tran is the transformed data, new var ended with _normalized is the normalized data from transformed data
	'''
	model_factors = ['new_ebitda_interest', 'new_net_margin', 'new_debt_bookcap', 'new_debt_ebitda']
	output = pd.DataFrame(columns = ['var_name', 'floor_value', 'cap_value', 'impute_value', 'mean_value', 'std_value'])
	for var_x in model_factors:
		select_col = indata.ix[:, var_x]
		select_col_wo99 = np.where(np.abs(select_col) == 100000, np.nan, select_col)			#1: assign -99999999 and 99999999 as missing
		select_col_wo99_wo_mis = select_col_wo99[~np.isnan(select_col_wo99)]					
		floor_cap = np.percentile(select_col_wo99_wo_mis, [min_c, max_c])						#2: calculate 5/95 percentile from non-missing data	
		select_col_after_fc = np.where(select_col <= floor_cap[0], floor_cap[0], np.where(select_col >= floor_cap[1], floor_cap[1], select_col))	#3: floor/cap data
		select_col_after_fc_wo_missing = select_col_after_fc[~np.isnan(select_col_after_fc)]
		if var_x in ['new_debt_bookcap', 'new_debt_ebitda']:
			impute_value = np.percentile(select_col_after_fc_wo_missing, 75)
		else:
			impute_value = np.percentile(select_col_after_fc_wo_missing, 25)
		select_col_after_fc_impute = np.where(np.isnan(select_col_after_fc), impute_value, select_col_after_fc)
		mean_value = np.mean(select_col_after_fc_impute)
		std_value = np.std(select_col_after_fc_impute)
		select_col_after_fc_impute_normalized = (select_col_after_fc_impute - mean_value) / std_value
		indata[var_x+'_tran'] = select_col_after_fc_impute
		indata[var_x+'_normalized'] = select_col_after_fc_impute_normalized
		summarized_data = pd.DataFrame([var_x, floor_cap[0], floor_cap[1], impute_value, mean_value, std_value]).T
		summarized_data.columns = ['var_name', 'floor_value', 'cap_value', 'impute_value', 'mean_value', 'std_value']
		output = pd.concat([output, summarized_data], axis = 0, ignore_index = True) 
	return output	

# cap/floor and impute missing on the training data
normalized_summary_matrix = normalize_train(data_after_pw,  min_c = 5, max_c = 95)
	
def normalize_test(indata, coef_matrix):
	model_factors = ['new_ebitda_interest', 'new_net_margin', 'new_debt_bookcap', 'new_debt_ebitda']
	coef_matrix2 = coef_matrix.set_index('var_name')
	for var_x in model_factors:
		floor_value, cap_value, impute_value, mean_value, std_value = coef_matrix2.ix[var_x, ['floor_value', 'cap_value', 'impute_value', 'mean_value', 'std_value']]
		select_col = indata.ix[:, var_x]																										#select column	
		select_col = np.where(select_col == -100000, floor_value, np.where(select_col == 100000, cap_value, select_col))					# replace -99999999/99999999
		select_col_after_fc = np.where(select_col <= floor_value, floor_value, np.where(select_col >= cap_value, cap_value, select_col))		#floor cap
		select_col_after_fc_impute = np.where(np.isnan(select_col_after_fc), impute_value, select_col_after_fc)									#impute		
		select_col_after_fc_impute_normalized = (select_col_after_fc_impute - mean_value) / std_value
		indata[var_x+'_tran'] = select_col_after_fc_impute
		indata[var_x+'_normalized'] = select_col_after_fc_impute_normalized
	tran_vars = [x for x in list(indata) if '_tran' in x]
	normalized_vars = [x for x in list(indata) if '_normalized' in x]
	return indata.ix[:, tran_vars + normalized_vars].describe()

	
f = 'frr_rank ~ new_ebitda_interest_tran + new_net_margin_tran + new_debt_bookcap_tran + new_debt_ebitda_tran'
lm = smf.ols(formula = str(f), data = data_after_pw).fit()
print lm.summary()

data_after_pw['pred_lm'] = map(round, lm.predict(data_after_pw))
print pd.crosstab(data_after_pw['pred_lm'], data_after_pw['frr_rank'], margins = True).to_string()



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




###############################################################################################################################################	
for i in range(2008, 2015):
    output = lcm_dataprep(i, i+1)
    output.to_excel(r'R:\Global_Market_Risk\RCST\WCM\Huiming_Song\LCM\lcm_after_pw_' + str(i) + '.xlsx', index = False)
	
# lcm2008 = lcm_dataprep(2008, 2009)		
# lcm2009 = lcm_dataprep(2009, 2010)	
# lcm2010 = lcm_dataprep(2010, 2011)	
# lcm2011 = lcm_dataprep(2011, 2012)	
# lcm2012 = lcm_dataprep(2012, 2013)	
# lcm2013 = lcm_dataprep(2013, 2014)	
# lcm2014 = lcm_dataprep(2014, 2015)	

final_default = pd.DataFrame()
final_data = pd.DataFrame()
for i in range(2008, 2016):
	output = lcm_dataprep(i, i+1)
	final_data = pd.concat([final_data, output], axis = 0)
	final_default = pd.concat([final_default, output.query('default_flag == 1')], axis = 0)
    


quant_vars = ['entityuen', 'final_form_date', 'financial_statement_date', 'default_date', 'ebitda_interest', 'net_margin', 'debt_book', 'debt_ebitda']

reb_def.query('year == 2010').ix[:, ['entityuen', 'archive_id', 'financial_statement_date', 'final_form_date', 'default_date', 'pw']].sort('entityuen')
pd.merge(reb_def.query('year == 2010').ix[:, ['year', 'archive_id']], 

# 08 verified:   reb_def.query('year == 2009').sort('archive_id').archive_id == d08.sort('archive_id').archive_id  

# 1: find different
    # archive_id  entityuen
# 12       50066   23024896
# 15       72053      33347
# 33       56918   35013227
# 39      105818   35018332
reb_def.query('year == 2010').ix[~reb_def.query('year == 2010').archive_id.isin(d09.archive_id), ['archive_id', 'entityuen']] 

# 2: find one of different data, for example   33347
   # entityuen  archive_id  dura_attst_df  dura_fybegin_attst  pw
# 6      33347       48301      17.741637           -7.490914   2
# 4      33347       72053       9.429352            0.821372   4
# 3      33347       96691       2.496971            7.753753   4
# 5      33347       42598      21.881353          -11.630629   9
lcm_yr_data_sort.query('entityuen == 33347').ix[:, ['entityuen', 'archive_id', 'dura_attst_df', 'dura_fybegin_attst', 'pw']]

 