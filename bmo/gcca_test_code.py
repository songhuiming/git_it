##################################################################### Model Construction (REG directly)  #############################################################
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import scipy.stats as stats
from sklearn import metrics

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
	
def gini2(x, y):
	fpr, tpr, thresholds = metrics.roc_curve(y, x)
	return 2 * float(metrics.auc(fpr, tpr)) - 1
	
model_factors = [x.lower() for x in ['NET_MRGN_RTO', 'DEBT_SRVC_COV_RTO', 'YRS_IN_BUSINESS_b', 'debt_to_ebitda_rto', 'cur_rto']]

gcca_dev = pd.read_csv(r'H:\work\onRequest\gc_ca_run\Dev_Tr2.csv')
gcca_dev.columns = [x.lower() for x in gcca_dev.columns]

gcca_test = pd.read_csv(r'H:\work\onRequest\gc_ca_run\OOT_Tr2.csv')
gcca_test.columns = [x.lower() for x in gcca_test.columns]

def normalize_train(indata, min_c = 5, max_c = 95):
	'''
	1: assign -99999 and 99999 as missing  
	2: calculate 5/95 percentile from non-missing data
	3: floor / cap the data  (dev + 12 13 14 sample, )
	4: impute missing by the worst side 3rd quantile (or median)
	5: replace -99999999 or 99999999 by min/max, that is, the 5/95 percentile above 
	6: output: 'output' is data summary info('floor_value', 'cap_value', 'impute_value', 'mean_value', 'std_value') for transform, and the values to use on test data
	7: the new variable ended with _tran is the transformed data, new var ended with _normalized is the normalized data from transformed data
	'''
	model_factors = [x.lower() for x in ['NET_MRGN_RTO', 'DEBT_SRVC_COV_RTO', 'YRS_IN_BUSINESS_b', 'debt_to_ebitda_rto', 'cur_rto']]
	output = pd.DataFrame(columns = ['var_name', 'floor_value', 'cap_value', 'impute_value', 'mean_value', 'std_value'])
	for var_x in model_factors:
		select_col = indata.ix[:, var_x]
		select_col_wo99 = np.where(np.abs(select_col) == 99999, np.nan, select_col)			#1: assign -99999 and 99999 as missing
		select_col_wo99_wo_mis = select_col_wo99[~np.isnan(select_col_wo99)]					
		floor_cap = np.percentile(select_col_wo99_wo_mis, [min_c, max_c])						#2: calculate 5/95 percentile from non-missing data	
		select_col_after_fc = np.where(select_col <= floor_cap[0], floor_cap[0], np.where(select_col >= floor_cap[1], floor_cap[1], select_col))	#3: floor/cap data
		select_col_after_fc_wo_missing = select_col_after_fc[~np.isnan(select_col_after_fc)]
		if var_x in ['debt_to_ebitda_rto']:
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

## applying the function to final_sample will add _tran and _normalized factors in the training data set, the output is information to be used in validation data	
normalized_summary_matrix = normalize_train(indata = gcca_dev)
	
def normalize_test(indata, coef_matrix):
	model_factors = [x.lower() for x in ['NET_MRGN_RTO', 'DEBT_SRVC_COV_RTO', 'YRS_IN_BUSINESS_b', 'debt_to_ebitda_rto', 'cur_rto']]
	coef_matrix2 = coef_matrix.set_index('var_name')
	for var_x in model_factors:
		floor_value, cap_value, impute_value, mean_value, std_value = coef_matrix2.ix[var_x, ['floor_value', 'cap_value', 'impute_value', 'mean_value', 'std_value']]
		select_col = indata.ix[:, var_x]																										#select column	
		select_col = np.where(select_col == -99999, floor_value, np.where(select_col == 99999, cap_value, select_col))					# replace -99999/99999
		select_col_after_fc = np.where(select_col <= floor_value, floor_value, np.where(select_col >= cap_value, cap_value, select_col))		#floor cap
		select_col_after_fc_impute = np.where(np.isnan(select_col_after_fc), impute_value, select_col_after_fc)									#impute		
		select_col_after_fc_impute_normalized = (select_col_after_fc_impute - mean_value) / std_value
		indata[var_x+'_tran'] = select_col_after_fc_impute
		indata[var_x+'_normalized'] = select_col_after_fc_impute_normalized
	tran_vars = [x for x in list(indata) if '_tran' in x]
	normalized_vars = [x for x in list(indata) if '_normalized' in x]
	return indata.ix[:, tran_vars + normalized_vars].describe()

## applying the function to f121314 to do transformation and normalization
gcca_test_tran_summary = normalize_test(gcca_test, normalized_summary_matrix)
 
 
tran_vars = [x + '_tran' for x in model_factors] 
f = 'df ~ ' + ' + '.join(tran_vars)

f = 'df ~ net_mrgn_rto_tran + debt_srvc_cov_rto_tran + C(yrs_in_business_b) + debt_to_ebitda_rto_tran + cur_rto_tran'
m1 = smf.logit(formula = str(f), data = gcca_dev).fit()


	
print auc(m1.predict(gcca_dev), gcca_dev.df) * 2 - 1
print auc(m1.predict(gcca_test), gcca_test.df) * 2 - 1



	
	 



