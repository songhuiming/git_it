##################################################################### Model Construction (REG directly)  #############################################################
import statsmodels.formula.api as smf

model_factors = ['tot_sales_amt', 'dsc', 'yrs_in_bus', 'debt_to_tnw_rto', 'debt_to_ebitda_rto', 'net_margin_rto', 'cur_rto']

def normal_var(var_x, indata, min_c = 10, max_c = 90):
	'''
	# use the worse end 25% percentile to impute the missing value
	# step 1: calculate the percentiles by removing the min(-99999999) and max(99999999)
	# step 2: use this mean to impute missing values
	# step 3: floor and cap at 5% and 95% percentile
	# why impute first then floor and cap? dsc has lots of missing, if cap first, 95% pct will be nan
	'''
	select_col = indata.ix[:, var_x]
	if var_x in ['debt_to_ebitda_rto', 'debt_to_tnw_rto']:
		impute_value = np.percentile(indata.ix[(select_col != 99999999) & (select_col != -99999999), var_x], 75)
	else:
		impute_value = np.percentile(indata.ix[(select_col != 99999999) & (select_col != -99999999), var_x], 25)
	select_col = np.where(select_col.isnull(), impute_value, select_col)
	floor_cap = np.percentile(select_col, [min_c, max_c])
	select_col = np.where(select_col < floor_cap[0], floor_cap[0], np.where(select_col > floor_cap[1], floor_cap[1], select_col))
	# print "mean is %s, floor is %s, cap is %s" %(impute_value, floor_cap[0], floor_cap[1])
	select_col_std = (select_col - select_col.mean()) / select_col.std()
	return select_col_std

##############        normalize each factor	     ########################
for i in model_factors:
	final_sample[i+'_tran'] = normal_var(i, final_sample, 5, 95)

# single variable analysis for transformed factor by gini coefficient 
for i in ' dsc_tran + tot_sales_amt_tran + yrs_in_bus_tran + debt_to_ebitda_rto_tran + debt_to_tnw_rto_tran + cur_rto_tran + net_margin_rto'.split('+'):
	j = i.replace('_tran', '').strip()
	if j in ['debt_to_ebitda_rto', 'debt_to_tnw_rto']:
		auc_value = auc(final_sample.ix[:, j], final_sample.default_flag)
	else:
		auc_value = auc(-final_sample.ix[:, j], final_sample.default_flag)
	gini_coeff = 2 * auc_value - 1
	print "The Gini Coefficient for transformed factor %s is %s" %(i, gini_coeff)
	
# be careful: if standarize data, then here should not use 
final_sample['yrs_in_bus_bin'] = np.where(final_sample.yrs_in_bus < 10.761644, -0.430399, np.where(final_sample.yrs_in_bus < 17.901370, -0.019763, np.where(final_sample.yrs_in_bus < 30.821918, 0.082585, 0.524479)))
final_sample['yrs_in_bus_tran_bin'] = np.where(final_sample.yrs_in_bus_tran < -0.758049, -0.430399, np.where(final_sample.yrs_in_bus < -0.295956, -0.019763, np.where(final_sample.yrs_in_bus < 0.540280, 0.082585, 0.524479)))
	
regf1 = 'default_flag ~ dsc_tran + tot_sales_amt_tran + yrs_in_bus_tran + debt_to_ebitda_rto_tran + debt_to_tnw_rto_tran + cur_rto_tran'
regf2 = 'default_flag ~ dsc_tran + tot_sales_amt_tran + debt_to_ebitda_rto_tran + debt_to_tnw_rto_tran + cur_rto_tran + net_margin_rto_tran'	
regf3 = 'default_flag ~ dsc_tran + tot_sales_amt_tran + yrs_in_bus_tran + debt_to_ebitda_rto_tran + net_margin_rto_tran + cur_rto_tran'   # need some transformation
regf4 = 'default_flag ~ dsc_tran + tot_sales_amt_tran + _yrs_in_b + debt_to_ebitda_rto_tran + net_margin_rto_tran + cur_rto_tran'   # use woe transform for yrs_in_bus
regf5 = 'default_flag ~ dsc_tran + C(_tot_sales2) + yrs_in_bus_tran_bin + debt_to_ebitda_rto_tran + net_margin_rto_tran + cur_rto_tran'   # use woe transform for yrs_in_bus

# final sample exclude s2014 data
final_sample_til2013 = final_sample.query('yeartype != "2014MI" & yeartype != "2014HBC"')
final_sample_2014 = final_sample.query('yeartype == "2014MI" | yeartype == "2014HBC"')

regm1 = smf.logit(formula = str(regf1), data = final_sample_til2013).fit()
auc1 =  auc(regm1.predict(), final_sample_til2013.default_flag)
auc_preddata = auc(regm1.predict(final_sample_2014), final_sample_2014.default_flag)
print "The AUC for current model m1 is: %s, and AUC for OOT data is %s" %(auc1, auc_preddata)

regm2 = smf.logit(formula = str(regf2), data = final_sample_til2013).fit()
auc2 =  auc(regm2.predict(), final_sample_til2013.default_flag)
auc_preddata = auc(regm2.predict(final_sample_2014), final_sample_2014.default_flag)
print "The AUC for current model m2 is: %s, and AUC for OOT data is %s" %(auc2, auc_preddata)

regm3 = smf.logit(formula = str(regf3), data = final_sample_til2013).fit()
auc3 =  auc(regm3.predict(), final_sample_til2013.default_flag)
auc_preddata = auc(regm3.predict(final_sample_2014), final_sample_2014.default_flag)
print "The AUC for current model m3 is: %s, and AUC for OOT data is %s" %(auc3, auc_preddata)

regm4 = smf.logit(formula = str(regf4), data = final_sample_til2013).fit()
auc4 =  auc(regm4.predict(), final_sample_til2013.default_flag)
auc_preddata = auc(regm4.predict(final_sample_2014), final_sample_2014.default_flag)
print "The AUC for current model m4 is: %s, and AUC for OOT data is %s" %(auc4, auc_preddata)

regm5 = smf.logit(formula = str(regf5), data = final_sample_til2013).fit()
auc5 =  auc(regm5.predict(), final_sample_til2013.default_flag)
auc_preddata = auc(regm5.predict(final_sample_2014), final_sample_2014.default_flag)
print "The AUC for current model m5 is: %s, and AUC for OOT data is %s" %(auc5, auc_preddata)




