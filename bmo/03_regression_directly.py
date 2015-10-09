##################################################################### Model Construction (REG directly)  #############################################################
import statsmodels.formula.api as smf
import scipy.stats as stats

model_factors = ['tot_sales_amt', 'dsc', 'yrs_in_bus', 'debt_to_tnw_rto', 'debt_to_ebitda_rto', 'net_margin_rto', 'cur_rto']

def normalize_train(indata = final_sample, min_c = 5, max_c = 95):
	'''
	1: assign -99999999 and 99999999 as missing  
	2: calculate 5/95 percentile from non-missing data
	3: floor / cap the data  (dev + 12 13 14 sample, )
	4: impute missing by the worst side 3rd quantile (or median)
	5: replace -99999999 or 99999999 by min/max, that is, the 5/95 percentile above 
	6: output: 'output' is data summary info('floor_value', 'cap_value', 'impute_value', 'mean_value', 'std_value') for transform, and the values to use on test data
	7: the new variable ended with _tran is the transformed data, new var ended with _normalized is the normalized data from transformed data
	'''
	model_factors = ['tot_sales_amt', 'dsc', 'yrs_in_bus', 'debt_to_tnw_rto', 'debt_to_ebitda_rto', 'net_margin_rto', 'cur_rto']
	output = pd.DataFrame(columns = ['var_name', 'floor_value', 'cap_value', 'impute_value', 'mean_value', 'std_value'])
	for var_x in model_factors:
		select_col = indata.ix[:, var_x]
		select_col_wo99 = np.where(np.abs(select_col) == 99999999, np.nan, select_col)			#1: assign -99999999 and 99999999 as missing
		select_col_wo99_wo_mis = select_col_wo99[~np.isnan(select_col_wo99)]					
		floor_cap = np.percentile(select_col_wo99_wo_mis, [min_c, max_c])						#2: calculate 5/95 percentile from non-missing data	
		select_col_after_fc = np.where(select_col <= floor_cap[0], floor_cap[0], np.where(select_col >= floor_cap[1], floor_cap[1], select_col))	#3: floor/cap data
		select_col_after_fc_wo_missing = select_col_after_fc[~np.isnan(select_col_after_fc)]
		if var_x in ['debt_to_ebitda_rto', 'debt_to_tnw_rto']:
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
normalized_summary_matrix = normalize_train(final_sample)
	
def normalize_test(indata, coef_matrix):
	model_factors = ['tot_sales_amt', 'dsc', 'yrs_in_bus', 'debt_to_tnw_rto', 'debt_to_ebitda_rto', 'net_margin_rto', 'cur_rto']
	coef_matrix2 = coef_matrix.set_index('var_name')
	for var_x in model_factors:
		floor_value, cap_value, impute_value, mean_value, std_value = coef_matrix2.ix[var_x, ['floor_value', 'cap_value', 'impute_value', 'mean_value', 'std_value']]
		select_col = indata.ix[:, var_x]																										#select column	
		select_col = np.where(select_col == -99999999, floor_value, np.where(select_col == 99999999, cap_value, select_col))					# replace -99999999/99999999
		select_col_after_fc = np.where(select_col <= floor_value, floor_value, np.where(select_col >= cap_value, cap_value, select_col))		#floor cap
		select_col_after_fc_impute = np.where(np.isnan(select_col_after_fc), impute_value, select_col_after_fc)									#impute		
		select_col_after_fc_impute_normalized = (select_col_after_fc_impute - mean_value) / std_value
		indata[var_x+'_tran'] = select_col_after_fc_impute
		indata[var_x+'_normalized'] = select_col_after_fc_impute_normalized
	tran_vars = [x for x in list(indata) if '_tran' in x]
	normalized_vars = [x for x in list(indata) if '_normalized' in x]
	return indata.ix[:, tran_vars + normalized_vars].describe()

## applying the function to f121314 to do transformation and normalization
f121314_tran_summary = normalize_test(f121314, normalized_summary_matrix)

## transformation for years in business and total sales: to bin them based on the new bin cutoff points	
def yib_bin(indata, varx):
	select_col = indata.ix[:, varx]
	res = np.where(select_col < 10.333333, -0.413552, np.where(select_col < 17.901370, -0.042214, np.where(select_col < 30.666667, 0.079084, 0.521143)))
	return res
	
final_sample['yrs_in_bus_bin'] = yib_bin(final_sample, 'yrs_in_bus') 
f121314['yrs_in_bus_bin'] = yib_bin(f121314, 'yrs_in_bus')  
final_sample['yrs_in_bus_tran_bin'] = yib_bin(final_sample, 'yrs_in_bus_tran')  
f121314['yrs_in_bus_tran_bin'] = yib_bin(f121314, 'yrs_in_bus_tran')   
final_sample[u'_tot_sales2'] = np.where(final_sample.tot_sales_amt > 20000000, 3, np.where(final_sample.tot_sales_amt > 5000000, 2, 1))
f121314[u'_tot_sales2'] = np.where(f121314.tot_sales_amt > 20000000, 3, np.where(f121314.tot_sales_amt > 5000000, 2, 1))	
	
# Analysis 2: calculate the drivers median on each FRR	
model_factors_tran = ' dsc_tran + tot_sales_amt_tran + yrs_in_bus_tran + debt_to_ebitda_rto_tran + debt_to_tnw_rto_tran + cur_rto_tran + net_margin_rto_tran'.replace(" ", "").split('+')
print final_sample.ix[:, model_factors_tran + ['final_ranking']].groupby('final_ranking').median().to_string()

model_factors_norm = [x for x in list(final_sample) if "_normalized" in x]
print final_sample.ix[:, model_factors_norm + ['final_ranking']].groupby('final_ranking').median().to_string()

# frr_xtran_median = pd.DataFrame(columns = model_factors_tran)
# for i in model_factors_tran:
	# print '=' * 20 + ' ' + i + ' ' + '=' * 20
	# res = final_sample.ix[:, ['final_ranking', i]].groupby('final_ranking').median().reset_index().ix[:, 1]
	# print res
	# frr_xtran_median[i] = res
	# print frr_xtran_median.to_string()
	
## correlation between Drivers vs Final_ranking, skew and kurtosis
fs121314 = final_sample.query('yeartype != "2011Before" & final_ranking > 0')   			# no FRR for 2011 before data, and for two obs after 2011
for i in ' dsc_tran + tot_sales_amt_tran + yrs_in_bus_tran + debt_to_ebitda_rto_tran + debt_to_tnw_rto_tran + cur_rto_tran + net_margin_rto_tran'.replace(" ", "").split('+'):
	print "skew for %s is %s " %(i, stats.skew(final_sample.ix[:, i]))
	print "Kurt for %s is %s " %(i, stats.kurtosis(final_sample.ix[:, i]))
	print "Pearson Corr for %s is %s " %(i, stats.pearsonr(fs121314.ix[:, i], fs121314.final_ranking)[0])
	print "Spearman Corr for %s is %s " %(i, stats.spearmanr(fs121314.ix[:, i], fs121314.final_ranking)[0])
	
	
# single variable analysis for transformed factor by gini coefficient 
for i in ' dsc_tran + tot_sales_amt_tran + yrs_in_bus_tran + debt_to_ebitda_rto_tran + debt_to_tnw_rto_tran + cur_rto_tran + net_margin_rto_tran'.replace(" ", "").split('+'):
	if i in ['debt_to_ebitda_rto_tran', 'debt_to_tnw_rto_tran']:
		auc_value = auc(final_sample.ix[:, i], final_sample.default_flag)
	else:
		auc_value = auc(-final_sample.ix[:, i], final_sample.default_flag)
	gini_coeff = 2 * auc_value - 1
	print "The Gini Coefficient for transformed factor %s is %s" %(i, gini_coeff)
 
	
regf1 = 'default_flag ~ dsc_tran + tot_sales_amt_tran + yrs_in_bus_tran + debt_to_ebitda_rto_tran + debt_to_tnw_rto_tran + cur_rto_tran'
regf2 = 'default_flag ~ dsc_tran + tot_sales_amt_tran + yrs_in_bus_tran + debt_to_ebitda_rto_tran + debt_to_tnw_rto_tran + net_margin_rto_tran'
regf3 = 'default_flag ~ dsc_tran + tot_sales_amt_tran + debt_to_ebitda_rto_tran + debt_to_tnw_rto_tran + cur_rto_tran + net_margin_rto_tran'	
#regf4 = 'default_flag ~ dsc_tran + tot_sales_amt_tran + yrs_in_bus_tran + debt_to_ebitda_rto_tran + net_margin_rto_tran + cur_rto_tran'   # need some transformation
regf4 = 'default_flag ~ dsc_tran + tot_sales_amt_tran + yrs_in_bus_tran_bin + debt_to_ebitda_rto_tran + net_margin_rto_tran + cur_rto_tran'   # use woe transform for yrs_in_bus
regf5 = 'default_flag ~ dsc_tran + C(_tot_sales2) + yrs_in_bus_tran_bin + debt_to_ebitda_rto_tran + net_margin_rto_tran + cur_rto_tran'   # use woe transform for yrs_in_bus

# final sample exclude s2014 data

regm1 = smf.logit(formula = str(regf1), data = final_sample).fit()
auc1 =  auc(regm1.predict(), final_sample.default_flag)
auc_preddata = auc(regm1.predict(f121314), f121314.default_flag)
print "The AUC for current model m1 is: %s, and AUC for OOT data is %s" %(auc1, auc_preddata)

regm2 = smf.logit(formula = str(regf2), data = final_sample).fit()
auc2 =  auc(regm2.predict(), final_sample.default_flag)
auc_preddata = auc(regm2.predict(f121314), f121314.default_flag)
print "The AUC for current model m2 is: %s, and AUC for OOT data is %s" %(auc2, auc_preddata)

regm3 = smf.logit(formula = str(regf3), data = final_sample).fit()
auc3 =  auc(regm3.predict(), final_sample.default_flag)
auc_preddata = auc(regm3.predict(f121314), f121314.default_flag)
print "The AUC for current model m3 is: %s, and AUC for OOT data is %s" %(auc3, auc_preddata)

regm4 = smf.logit(formula = str(regf4), data = final_sample).fit()
auc4 =  auc(regm4.predict(), final_sample.default_flag)
auc_preddata = auc(regm4.predict(f121314), f121314.default_flag)
print "The AUC for current model m4 is: %s, and AUC for OOT data is %s" %(auc4, auc_preddata)

regm5 = smf.logit(formula = str(regf5), data = final_sample).fit()
auc5 =  auc(regm5.predict(), final_sample.default_flag)
auc_preddata = auc(regm5.predict(f121314), f121314.default_flag)
print "The AUC for current model m5 is: %s, and AUC for OOT data is %s" %(auc5, auc_preddata)


##### calibration

def pdc(pdm):
	drp = 0.0258					# long run pd
	drs = 283.0 / 1655.0 			# sample default rate
	return (drp/drs)*pdm / (drp/drs*pdm + (1-drp)/(1-drs)*(1-pdm))  


pdIntval = [0.0003, 0.0007, 0.0011, 0.0019, 0.0032, 0.0054, 0.0091, 0.0154, 0.0274, 0.0516, 0.097, 0.1823, 1]
msRating = ['I-2', 'I-3', 'I-4', 'I-5', 'I-6', 'I-7', 'S-1', 'S-2', 'S-3', 'S-4', 'P-1', 'P-2', 'P-3', 'D-1']
pcRR = [10, 15, 20, 25, 30, 35, 40, 45, 46, 47, 50, 51, 52]
ranking = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
newMapPD = [0.0003, 0.0005, 0.0009, 0.0015, 0.0026, 0.0043, 0.0073, 0.0123, 0.0214, 0.0395, 0.0743, 0.1397, 0.2412]
oldMapPD = [0.0004, 0.0007, 0.0013, 0.0025, 0.004, 0.0067, 0.0107, 0.018, 0.0293, 0.0467, 0.0933, 0.18, 0.3333]
 
def ranking_compare(r1, r2):
	if r1.shape[0] == r2.shape[0]:
		concat_data = pd.concat([r1, r2], join_axes = None, axis = 1, ignore_index = True)
		concat_data.columns = ['pred_ranking', 'final_ranking']
		concat_data['ranking_diff'] = abs(concat_data.pred_ranking - concat_data.final_ranking)
		concat_data['ranking_cons1'] = np.where(concat_data.ranking_diff == 0, 'Consistant', 'Different')
		concat_data['ranking_cons2'] = np.where(concat_data.pred_ranking > concat_data.final_ranking, "More Conservative", np.where(concat_data.pred_ranking < concat_data.final_ranking, "Less Conservative", 'Consistant')) 
		concat_data['ranking_cons3'] = np.where(concat_data.ranking_diff == 0, '0: Consistant', np.where(concat_data.ranking_diff <= 1, '1: within 1 notch', np.where(concat_data.ranking_diff <= 2, '2: within 2 notch', '3: 3+ Difference')))
		print concat_data.ranking_cons1.value_counts(dropna = False).sort_index()
		print '-'*20
		print concat_data.ranking_cons2.value_counts(dropna = False).sort_index()
		print '-'*20
		print concat_data.ranking_cons3.value_counts(dropna = False).sort_index()
		print '-'*20
		print pd.crosstab(concat_data.pred_ranking, concat_data.final_ranking, margins = True).to_string()
	else:
		print "r1 r2 Shape NOT Match"


# compare the predictive ranking from model v.s. final ranking for FS121314(sample) data
reg41 = pd.Series(np.digitize(pdc(regm5.predict(fs121314)), pdIntval) + 2)
reg42 = fs121314.final_ranking
ranking_compare(reg41, reg42)

# compare the predictive ranking from model v.s. final ranking for F121314(population) data
reg41 = pd.Series(np.digitize(pdc(regm5.predict(f121314)), pdIntval) + 2)
reg42 = f121314.final_ranking
ranking_compare(reg41, reg42)

# xByCoeff is x * coeff,    s1 = sum(x_i * coeff_i + intercept)     #the intercept is  -2.927 to adjust avg pred pd to 0.0258
regm3VarName = 'dsc_tran + tot_sales_amt_tran + debt_to_ebitda_rto_tran + debt_to_tnw_rto_tran + cur_rto_tran + net_margin_rto_tran'.replace('+', ' ').split()
def calibration2(indata = final_sample, vars = regm3VarName, fit_model = regm3, new_intercept = - 2.927):
	xByCoeff = indata.ix[:, vars] * fit_model.params[1:]			# s = Sigma_{x_i * beta_i}
	s1 = np.sum(xByCoeff, axis = 1) + fit_model.params[0]  		#this is for calculate the predicted value from model before calibration
	s2 = np.sum(xByCoeff, axis = 1) + new_intercept				#the new_intercept is used for calibration
	before_calib_pred = np.exp(s1) / (1 + np.exp(s1))
	after_calib_pred = np.exp(s2) / (1 + np.exp(s2))
	print "the average PD before calibration is: %s" %(np.mean(before_calib_pred))
	print "the average PD after calibration is: %s" %(np.mean(after_calib_pred))
	return after_calib_pred
	
final_sample_after_calib_pred = calibration2(indata = final_sample, vars = regm3VarName, fit_model = regm3, new_intercept = - 2.927)
fs121314_after_calib_pred = calibration2(indata = fs121314, vars = regm3VarName, fit_model = regm3, new_intercept = - 2.927)
f121314_after_calib_pred = calibration2(indata = f121314, vars = regm3VarName, fit_model = regm3, new_intercept = - 2.927)
	
ranking_compare(pd.Series(np.digitize(after_calib_pred, pdIntval) + 2), final_sample.final_ranking)
ranking_compare(pd.Series(np.digitize(fs121314_after_calib_pred, pdIntval) + 2), fs121314.final_ranking)
ranking_compare(pd.Series(np.digitize(f121314_after_calib_pred, pdIntval) + 2), f121314.final_ranking)

 
final_sample.to_excel("H:\\work\usgc\\2015\\quant\\2015_supp\\final_sample_dir_reg_m3.xlsx")





