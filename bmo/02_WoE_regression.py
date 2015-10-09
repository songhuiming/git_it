
##################################################################### Method 1: Model Construction (WOE)  #############################################################
import statsmodels.formula.api as smf
 
##############################################################################  01: Calculate WOE  #############################################################################

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

#7 net_margin_rto
net_margin_cut = [0, 0.01572, 0.1321]
net_margin_woe = [-0.98827, 0.16849, 0.62193, 1.03296]

def woe(x, bin_x, woe_value, right=1):
	return [woe_value[i] for i in np.digitize(np.array(x), np.array(bin_x), right = right)]

model_factors_woe = [x for x in list(final_sample) if x.startswith('_')]

###################     WoE 1: WoE transformation on the Sampled data    

def woe_bin(indata):
	#1: dsc  logic: dsc=na, _dsc=0; dsc=0, _dsc=-0.99743; the rest are right included; ebit<0, go to bad;
	indata['_dsc'] = woe(indata.ix[:, u'dsc'], dsc_cut, dsc_woe)
	indata[u'_dsc'][indata[u'dsc'].isnull()] = 0
	indata[u'_dsc'][indata[u'dsc'] == 0] = -0.99743
	indata[u'_dsc'][(indata[u'ebitda_amt'] < 0) & (indata[u'dsc'].isnull())] = -1.33433
	indata[u'_dsc'].value_counts(dropna = 0).sort_index()
		# dsc2 = [-inf, 0, 1, 1.25, 2.06452, 4.45, inf]
		# pd.value_counts(pd.cut(indata[u'_dsc'], dsc2, right = 1), sort=1, dropna = 0)	 
	#2: cur_rto:
	indata[u'_cur_rto'] = woe(indata.ix[:, u'cur_rto'], cur_rto_cut, cur_woe)
	indata[u'_cur_rto'][indata[u'cur_rto'].isnull()] = 0
	indata[u'_cur_rto'][indata[u'cur_rto'] == 0.87565] = -1.13540
	indata[u'_cur_rto'][(indata[u'cur_rto'].isnull()) & (indata[u'cur_liab_amt'] > 0) & (indata[u'cur_ast_amt'] <= 0)] = -1.13540 
	indata[u'_cur_rto'].value_counts().sort_index()
	#3: d_2_tnw=0: _d_2_tnw=1.32242; d_2_tnw=na & tnw<=0: _d_2_tnw=-1.06007  
	indata[u'_debt_2_tnw'] = woe(indata.ix[:, u'debt_to_tnw_rto'], debt_2_tnw, debt_2_tnw_woe)
	indata[u'_debt_2_tnw'][indata[u'debt_to_tnw_rto'].isnull()] = 0
	indata[u'_debt_2_tnw'][indata[u'debt_to_tnw_rto'] == 0] = 1.32242
	indata[u'_debt_2_tnw'][(indata[u'debt_to_tnw_rto'].isnull()) & (indata[u'tangible_net_worth_amt'] <=0)] = -1.06007 
	indata[u'_debt_2_tnw'].value_counts().sort_index()
	#4: yr_in_b = na = -1.11391, as sas code: if Yrs_In_B<0 then _Yrs_In_B=-1.11391;
	indata[u'_yrs_in_b'] = woe(indata.ix[:, u'yrs_in_bus'], yrs_in_b, yrs_in_b_woe)
	indata[u'_yrs_in_b'][indata[u'yrs_in_bus'].isnull()] = -1.11391
	indata[u'_yrs_in_b'][indata[u'yrs_in_bus'] == 3] = -1.11391
	indata[u'_yrs_in_b'][indata[u'yrs_in_bus'] < 0] = -1.11391
	indata[u'_yrs_in_b'].value_counts().sort_index()
	#5: left included in sas: low-<5000000   5000000-<20000000   20000000-High
	indata[u'_tot_sales'] = woe(indata[u'tot_sales_amt'], tot_sales, tot_sales_woe, right = 0)
	indata[u'_tot_sales'][indata[u'tot_sales_amt'].isnull()] = -0.56285;
	indata[u'_tot_sales'].value_counts().sort_index()
	# 2nd way to transform total sales: binned based on small busibess / business banking / mid-market threshold
	indata[u'_tot_sales2'] = np.where(indata.tot_sales_amt > 20000000, 3, np.where(indata.tot_sales_amt > 5000000, 2, 1))
	#6: debt_to_ebitda
	indata[u'_dt_2_ebitda'] = woe(indata[u'debt_to_ebitda_rto'], dt_2_ebitda, dt_2_ebitda_woe)
	indata[u'_dt_2_ebitda'][indata[u'debt_to_ebitda_rto'] == 0] = 1.40998
	indata[u'_dt_2_ebitda'][indata[u'debt_to_ebitda_rto'].isnull()] = 0
	indata[u'_dt_2_ebitda'][(indata[u'debt_to_ebitda_rto'].isnull()) & (indata[u'ebitda_amt'] <= 0)] = -1.27817
	indata[u'_dt_2_ebitda'].value_counts().sort_index()
	#7: net margin = net income / total sales
	indata[u'_net_margin_rto'] = woe(indata[u'net_margin_rto'], net_margin_cut, net_margin_woe)
	indata[u'_net_margin_rto'][indata[u'net_margin_rto'] == 0] = 0.16849
	indata[u'_net_margin_rto'][indata[u'net_margin_rto'].isnull()] = 0
	indata[u'_net_margin_rto'][(indata[u'net_margin_rto'].isnull()) & (indata[u'net_inc_amt'] < 0)] = -0.98827
	indata[u'_net_margin_rto'].value_counts().sort_index()	
	return indata

## use woe to bin the model drivers	on the sample data and whole data
final_sample = woe_bin(final_sample)
f121314 = woe_bin(f121314)					#f121314 is the final population data of F2012 ~ F2014, fs121314 is the final sample data  
devdata_woe_bin =  woe_bin(devdata)	

## verify and compare the result with SAS
def verify_on_woe(indata):
	output = pd.DataFrame(columns = ['var_name', 'counts', 'proportion', 'average pd'])
	model_var_woe = [x for x in list(indata) if x.startswith('_')]
	for i in model_var_woe:
		cnts = indata.ix[:, i].value_counts().sort_index()
		average_pd = indata.groupby(i).default_flag.mean()
		res = pd.DataFrame([cnts, cnts / indata.shape[0], average_pd]).T
		res.columns = ['counts', 'proportion', 'average pd']
		res['var_name'] = i
		output = pd.concat([output, res], axis = 0)
	output1 = output.reset_index().ix[:, ['var_name', 'counts', 'proportion', 'average pd', 'index']]
	return output1    #last column index is WoE in fact, how to rename it?

fs_verify = verify_on_woe(final_sample)	    # compare this with the output from SAS
	
###################     WoE 2: WoE transformation on the F2014 Data

f1 = 'default_flag ~ _dsc + _debt_2_tnw + _yrs_in_b + _tot_sales + _dt_2_ebitda + _cur_rto'				#by SFA, _yrs_in_b is predictive
f2 = 'default_flag ~ _dsc + _debt_2_tnw + _yrs_in_b + _tot_sales + _dt_2_ebitda + _net_margin_rto'
f3 = 'default_flag ~ _dsc + _yrs_in_b + _tot_sales + _dt_2_ebitda + _cur_rto + _net_margin_rto'			#f3 f4 has same cut pt, f3 use woe
f4 = 'default_flag ~ _dsc + _yrs_in_b + C(_tot_sales2) + _dt_2_ebitda + _cur_rto + _net_margin_rto'     #f4 use indicator for sales, split to sbc / bb / ...
f5 = 'default_flag ~ _dsc + _debt_2_tnw + _tot_sales + _dt_2_ebitda + _cur_rto + _net_margin_rto'

m1 = smf.logit(formula = str(f1), data = final_sample).fit()
auc1 = auc(m1.predict(), final_sample.default_flag)
auc_preddata = auc(m1.predict(f121314), f121314.default_flag)
print "The AUC for current model m1 is: %s, and AUC for OOT data is %s" %(auc1, auc_preddata)

m2 = smf.logit(formula = str(f2), data = final_sample).fit()
auc2 = auc(m2.predict(), final_sample.default_flag)
auc_preddata = auc(m2.predict(f121314), f121314.default_flag)
print "The AUC for current model m2 is: %s, and AUC for OOT data is %s" %(auc2, auc_preddata)

m3 = smf.logit(formula = str(f3), data = final_sample).fit()
auc3 = auc(m3.predict(), final_sample.default_flag)
auc_preddata = auc(m3.predict(f121314), f121314.default_flag)
print "The AUC for current model m3 is: %s, and AUC for OOT data is %s" %(auc3, auc_preddata)

m4 = smf.logit(formula = str(f4), data = final_sample).fit()
auc4 = auc(m4.predict(), final_sample.default_flag)
auc_preddata = auc(m4.predict(f121314), f121314.default_flag)
print "The AUC for current model m4 is: %s, and AUC for OOT data is %s" %(auc4, auc_preddata)

m5 = smf.logit(formula = str(f5), data = final_sample).fit()
auc5 = auc(m5.predict(), final_sample.default_flag)
auc_preddata = auc(m5.predict(f121314), f121314.default_flag)
print "The AUC for current model m5 is: %s, and AUC for OOT data is %s" %(auc5, auc_preddata)


###################################  Result on All Data with Final Ranking    #####################################
 
def pdc(pdm):						# calibration function
	drp = 0.0258					# long run pd
	drs = 283.0 / 1655.0 			# sample default rate
	return drp/drs*pdm / (drp/drs*pdm + (1-drp)/(1-drs)*(1-pdm))

def ranking_compare(r1, r2):
	print '=' * 25 + '  Compare Pred Ranking v.s. Final Ranking  ' + '=' * 25 
	if r1.shape[0] == r2.shape[0]:
		concat_data = pd.concat([r1, r2], join_axes = None, axis = 1, ignore_index = True)
		concat_data.columns = ['pred_ranking', 'final_ranking']
		concat_data['ranking_diff'] = abs(concat_data.pred_ranking - concat_data.final_ranking)
		concat_data['ranking_cons1'] = np.where(concat_data.ranking_diff == 0, 'Consistant', 'Different')
		concat_data['ranking_cons2'] = np.where(concat_data.pred_ranking > concat_data.final_ranking, "More Conservative", np.where(concat_data.pred_ranking < concat_data.final_ranking, "Less Conservative", 'Consistant')) 
		concat_data['ranking_cons3'] = np.where(concat_data.ranking_diff == 0, '0: Consistant', np.where(concat_data.ranking_diff == 1, '1: 1 notch diff', np.where(concat_data.ranking_diff == 2, '2: 2 notch diff', '3: 3+ different')))
		print concat_data.ranking_cons1.value_counts(dropna = False).sort_index()
		print '-'*20
		print concat_data.ranking_cons2.value_counts(dropna = False).sort_index()
		print '-'*20
		print concat_data.ranking_cons3.value_counts(dropna = False).sort_index()
		print '-'*20
		print pd.crosstab(concat_data.pred_ranking, concat_data.final_ranking, margins = True).to_string()
	else:
		print "r1 r2 Shape NOT Match"

pdIntval = [0.0003, 0.0007, 0.0011, 0.0019, 0.0032, 0.0054, 0.0091, 0.0154, 0.0274, 0.0516, 0.097, 0.1823, 1]
msRating = ['I-2', 'I-3', 'I-4', 'I-5', 'I-6', 'I-7', 'S-1', 'S-2', 'S-3', 'S-4', 'P-1', 'P-2', 'P-3', 'D-1']
pcRR = [10, 15, 20, 25, 30, 35, 40, 45, 46, 47, 50, 51, 52]
ranking = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
newMapPD = [0.0003, 0.0005, 0.0009, 0.0015, 0.0026, 0.0043, 0.0073, 0.0123, 0.0214, 0.0395, 0.0743, 0.1397, 0.2412]
oldMapPD = [0.0004, 0.0007, 0.0013, 0.0025, 0.004, 0.0067, 0.0107, 0.018, 0.0293, 0.0467, 0.0933, 0.18, 0.3333]		

# compare the predictive ranking from model v.s. final ranking for F2012 F2013 F2014 Sampled data
#f121314 is the final population data of F2012 ~ F2014, fs121314 is the final sample data  
fs121314 = final_sample.query('yeartype != "2011Before"').reset_index()
r41 = pd.Series(np.digitize(pdc(m4.predict(fs121314)), pdIntval) + 2)
r42 = fs121314.final_ranking
ranking_compare(r41, r42)

# compare the predictive ranking from model v.s. final ranking for F2012 F2013 F2014 Full data
r41all = pd.Series(np.digitize(pdc(m4.predict(f121314)), pdIntval) + 2)
r42all = f121314.final_ranking
ranking_compare(r41all, r42all) 

##############################################################################################
m1VarName = '_dsc + _debt_2_tnw + _yrs_in_b + _tot_sales + _dt_2_ebitda + _cur_rto'.replace('+', ' ').split()
def calibration2(indata = final_sample, vars = m1VarName, fit_model = m1, new_intercept = - 2.927):
	xByCoeff = indata.ix[:, vars] * fit_model.params[1:]			# s = Sigma_{x_i * beta_i}
	s1 = np.sum(xByCoeff, axis = 1) + fit_model.params[0]  		#this is for calculate the predicted value from model before calibration
	s2 = np.sum(xByCoeff, axis = 1) + new_intercept				#the new_intercept is used for calibration
	before_calib_pred = np.exp(s1) / (1 + np.exp(s1))
	after_calib_pred = np.exp(s2) / (1 + np.exp(s2))
	print "the average PD before calibration is: %s" %(np.mean(before_calib_pred))
	print "the average PD after calibration is: %s" %(np.mean(after_calib_pred))
	return after_calib_pred
	
final_sample_after_calib_pred = calibration2(indata = final_sample, vars = m1VarName, fit_model = m1, new_intercept = -3.93395)
fs121314_after_calib_pred = calibration2(indata = fs121314, vars = m1VarName, fit_model = m1, new_intercept = -3.93395)
f121314_after_calib_pred = calibration2(indata = f121314, vars = m1VarName, fit_model = m1, new_intercept = -3.93395)
	
ranking_compare(pd.Series(np.digitize(after_calib_pred, pdIntval) + 2), final_sample.final_ranking)
ranking_compare(pd.Series(np.digitize(fs121314_after_calib_pred, pdIntval) + 2), fs121314.final_ranking)
ranking_compare(pd.Series(np.digitize(f121314_after_calib_pred, pdIntval) + 2), f121314.final_ranking)

	

