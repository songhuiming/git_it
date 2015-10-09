import pandas as pd
import numpy as np
from datetime import datetime

lcmfile = pd.read_csv(r'R:\Global_Market_Risk\RCST\WCM\Rebecca Sun\LCM\Data for Huiming\lcm_data.csv')
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


def lcm_dataprep(pwYear, defaultYear):
	lcm_yr_data = lcmfile.ix[lcmfile.pw_year == pwYear, :]
	def_data = deffile.ix[deffile.pw_default_year <= defaultYear, :]
	lcm_yr_data = pd.merge(lcm_yr_data, def_data, on = 'entityuen', how = 'left')
	lcm_yr_data['default_flag'] = np.where(lcm_yr_data.default_date > 0, 1, 0)

	################################################################ calculate time interval, Performance Window   ######################################
	 
	lcm_yr_data['fyBegin'] = datetime.strptime(r'11/1/'+str(pwYear), '%m/%d/%Y')
	# delete all default obligors that default date is before the fiscal year start date
	lcm_yr_data = lcm_yr_data.ix[~((lcm_yr_data.default_date < lcm_yr_data.fyBegin) & (lcm_yr_data.default_date > 0)), :]
	
	lcm_yr_data['dura_fybegin_attst'] = (lcm_yr_data.ix[:, u'final_form_date'] - lcm_yr_data.ix[:, u'fyBegin']) / np.timedelta64(1, 'M')
	lcm_yr_data['dura_fs_attst'] = (lcm_yr_data.ix[:, u'final_form_date'] - lcm_yr_data.ix[:, u'financial_statement_date']) / np.timedelta64(1, 'M')
	lcm_yr_data['dura_fybegin_attst_abs'] = abs(lcm_yr_data.ix[:, 'dura_fybegin_attst'])
	lcm_yr_data['dura_attst_df'] = (lcm_yr_data.ix[:, u'default_date'] - lcm_yr_data.ix[:, u'final_form_date']) / np.timedelta64(1, 'M')
	lcm_yr_data['dura_fs_df'] = (lcm_yr_data.ix[:, u'default_date'] - lcm_yr_data.ix[:, u'financial_statement_date']) / np.timedelta64(1, 'M')

	lcm_yr_data.ix[:, 'pw'] = 0

	lcm_yr_data.ix[(lcm_yr_data['default_flag'] == 0) & ((-9 <= lcm_yr_data['dura_fybegin_attst']) & (lcm_yr_data['dura_fybegin_attst'] <0)), 'pw'] = 1     
	lcm_yr_data.ix[(lcm_yr_data['default_flag'] == 0) & ((0 <= lcm_yr_data['dura_fybegin_attst']) & (lcm_yr_data['dura_fybegin_attst'] <= 3)), 'pw'] = 2
	
	lcm_yr_data.ix[(lcm_yr_data['default_flag'] == 1) & (-12 <= lcm_yr_data['dura_fybegin_attst']) & (lcm_yr_data['dura_fybegin_attst'] <= 0) & ((3 <= lcm_yr_data['dura_attst_df']) & (lcm_yr_data['dura_attst_df'] <= 12)), 'pw'] = 1
	lcm_yr_data.ix[(lcm_yr_data['default_flag'] == 1) & (-12 <= lcm_yr_data['dura_fybegin_attst']) & (lcm_yr_data['dura_fybegin_attst'] <= 0) & ((12 < lcm_yr_data['dura_attst_df']) & (lcm_yr_data['dura_attst_df'] <= 18)), 'pw'] = 2
	lcm_yr_data.ix[(lcm_yr_data['default_flag'] == 1) & (-12 <= lcm_yr_data['dura_fybegin_attst']) & (lcm_yr_data['dura_fybegin_attst'] <= 0) & ((0 <= lcm_yr_data['dura_attst_df']) & (lcm_yr_data['dura_attst_df'] < 3)), 'pw'] = 3
	lcm_yr_data.ix[(lcm_yr_data['default_flag'] == 1) & (-12 <= lcm_yr_data['dura_fybegin_attst']) & (lcm_yr_data['dura_fybegin_attst'] <= 0) & ((0 > lcm_yr_data['dura_attst_df']) | (lcm_yr_data['dura_attst_df'] > 18)), 'pw'] = 5
	
	lcm_yr_data.ix[(lcm_yr_data['default_flag'] == 1) & ((0 <= lcm_yr_data['dura_attst_df']) & (lcm_yr_data['dura_fybegin_attst'] >= 0)), 'pw'] = 4
	lcm_yr_data.ix[(lcm_yr_data['default_flag'] == 1) & (-12 > lcm_yr_data['dura_fybegin_attst']), 'pw'] = 5
	
	lcm_yr_data.ix[((lcm_yr_data['default_flag'] == 0) & (lcm_yr_data['dura_fs_attst'] > 15)) | ((lcm_yr_data['default_flag'] == 1) & ((lcm_yr_data['dura_attst_df'] <= 0) | (lcm_yr_data['dura_fs_df'] > 24))), 'pw'] = 9

	lcm_yr_data['mkey'] = lcm_yr_data.ix[:, u'entityuen'].map(str) + lcm_yr_data.ix[:, u'pw_year'].map(str)
		
	lcm_yr_data_sort = lcm_yr_data.sort(['mkey', 'default_flag', 'pw', 'dura_fybegin_attst_abs'], ascending = [True, False, True, True])
	lcm_yr_data_after_pw = lcm_yr_data_sort.ix[((lcm_yr_data_sort['pw'] != 0) & (lcm_yr_data_sort['pw'] != 9)), :].drop_duplicates(['mkey']) 
	lcm_yr_data_after_pw['pw_year'] = lcm_yr_data_after_pw['pw_year'] + 1
	return lcm_yr_data_after_pw


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

