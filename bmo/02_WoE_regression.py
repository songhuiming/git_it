
##################################################################### Model Construction (WOE)  #############################################################
import statsmodels.formula.api as smf

# final sample exclude s2014 data
final_sample_til2013 = final_sample.query('yeartype != "2014MI" & yeartype != "2014HBC"')
final_sample_2014 = final_sample.query('yeartype == "2014MI" | yeartype == "2014HBC"')

f1 = 'default_flag ~ _dsc + _debt_2_tnw + _yrs_in_b + _tot_sales + _dt_2_ebitda + _cur_rto'
f2 = 'default_flag ~ _dsc + _debt_2_tnw + _yrs_in_b + _tot_sales + _dt_2_ebitda + _net_margin_rto'
f3 = 'default_flag ~ _dsc + _yrs_in_b + _tot_sales + _dt_2_ebitda + _cur_rto + _net_margin_rto'
f4 = 'default_flag ~ _dsc + _yrs_in_b + C(_tot_sales2) + _dt_2_ebitda + _cur_rto + _net_margin_rto'     # split to sbc / bb / ...

m1 = smf.logit(formula = str(f1), data = final_sample_til2013).fit()
#print m1.summary()
auc1 = auc(m1.predict(), final_sample_til2013.default_flag)
auc_preddata = auc(m1.predict(final_sample_2014), final_sample_2014.default_flag)
print "The AUC for current model m1 is: %s, and AUC for OOT data is %s" %(auc1, auc_preddata)

m2 = smf.logit(formula = str(f2), data = final_sample_til2013).fit()
auc2 = auc(m2.predict(), final_sample_til2013.default_flag)
auc_preddata = auc(m2.predict(final_sample_2014), final_sample_2014.default_flag)
print "The AUC for current model m2 is: %s, and AUC for OOT data is %s" %(auc2, auc_preddata)

m3 = smf.logit(formula = str(f3), data = final_sample_til2013).fit()
auc3 = auc(m3.predict(), final_sample_til2013.default_flag)
auc_preddata = auc(m3.predict(final_sample_2014), final_sample_2014.default_flag)
print "The AUC for current model m3 is: %s, and AUC for OOT data is %s" %(auc3, auc_preddata)

m4 = smf.logit(formula = str(f4), data = final_sample_til2013).fit()
auc4 = auc(m4.predict(), final_sample_til2013.default_flag)
auc_preddata = auc(m4.predict(final_sample_2014), final_sample_2014.default_flag)
print "The AUC for current model m4 is: %s, and AUC for OOT data is %s" %(auc4, auc_preddata)

