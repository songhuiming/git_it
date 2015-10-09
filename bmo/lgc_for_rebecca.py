import pandas as pd
import numpy as np
from datetime import datetime

lcmfile = pd.read_csv(r'R:\Global_Market_Risk\RCST\WCM\Rebecca Sun\LCM\Data for Huiming\lcm_data.csv')
deffile = pd.read_csv(r'R:\Global_Market_Risk\RCST\WCM\Rebecca Sun\LCM\Data for Huiming\default_list.csv')
comp = pd.read_csv(r'R:\Global_Market_Risk\RCST\WCM\Rebecca Sun\LCM\Data for Huiming\compare PW\PW2010_CLEAN.csv')

lcmfile.columns = [x.lower() for x in lcmfile.columns]
deffile.columns = [x.lower() for x in deffile.columns]

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



lcm2009 = lcmfile.ix[lcmfile.pw_year == 2009, :]
def2010 = deffile.ix[deffile.pw_default_year <= 2010, :]
lcm2009 = pd.merge(lcm2009, def2010, on = 'entityuen', how = 'left')
lcm2009['default_flag'] = np.where(lcm2009.default_date > 0, 1, 0)

################################################################ calculate time interval, Performance Window for 2014 ######################################
 
lcm2009['fyBegin'] = datetime.strptime(r'11/1/2009', '%m/%d/%Y')
lcm2009 = lcm2009.ix[~((lcm2009.default_date < lcm2009.fyBegin) & (lcm2009.default_date > 0)), :]
lcm2009['dura_fybegin_attst'] = (lcm2009.ix[:, u'final_form_date'] - lcm2009.ix[:, u'fyBegin']) / np.timedelta64(1, 'M')
lcm2009['dura_fs_attst'] = (lcm2009.ix[:, u'final_form_date'] - lcm2009.ix[:, u'financial_statement_date']) / np.timedelta64(1, 'M')
lcm2009['dura_fybegin_attst_abs'] = abs(lcm2009.ix[:, 'dura_fybegin_attst'])
lcm2009['dura_attst_df'] = (lcm2009.ix[:, u'default_date'] - lcm2009.ix[:, u'final_form_date']) / np.timedelta64(1, 'M')
lcm2009['dura_fs_df'] = (lcm2009.ix[:, u'default_date'] - lcm2009.ix[:, u'financial_statement_date']) / np.timedelta64(1, 'M')

lcm2009.ix[:, 'pw'] = 0

lcm2009.ix[(lcm2009['default_flag'] == 0) & ((-9 <= lcm2009['dura_fybegin_attst']) & (lcm2009['dura_fybegin_attst'] <0)), 'pw'] = 1     
lcm2009.ix[(lcm2009['default_flag'] == 0) & ((0 <= lcm2009['dura_fybegin_attst']) & (lcm2009['dura_fybegin_attst'] <= 3)), 'pw'] = 2

lcm2009.ix[(lcm2009['default_flag'] == 1) & (-12 <= lcm2009['dura_fybegin_attst']) & (lcm2009['dura_fybegin_attst'] <= 0) & ((3 <= lcm2009['dura_attst_df']) & (lcm2009['dura_attst_df'] <= 12)), 'pw'] = 1
lcm2009.ix[(lcm2009['default_flag'] == 1) & (-12 <= lcm2009['dura_fybegin_attst']) & (lcm2009['dura_fybegin_attst'] <= 0) & ((12 < lcm2009['dura_attst_df']) & (lcm2009['dura_attst_df'] <= 18)), 'pw'] = 2
lcm2009.ix[(lcm2009['default_flag'] == 1) & (-12 <= lcm2009['dura_fybegin_attst']) & (lcm2009['dura_fybegin_attst'] <= 0) & ((0 <= lcm2009['dura_attst_df']) & (lcm2009['dura_attst_df'] < 3)), 'pw'] = 3
lcm2009.ix[(lcm2009['default_flag'] == 1) & (-12 <= lcm2009['dura_fybegin_attst']) & (lcm2009['dura_fybegin_attst'] <= 0) & ((0 > lcm2009['dura_attst_df']) | (lcm2009['dura_attst_df'] > 18)), 'pw'] = 5

lcm2009.ix[((lcm2009['default_flag'] == 1) & (0 <= lcm2009['dura_attst_df']) & (lcm2009['dura_fybegin_attst'] >= 0)), 'pw'] = 4
lcm2009.ix[(lcm2009['default_flag'] == 1) & (-12 > lcm2009['dura_fybegin_attst']), 'pw'] = 5

lcm2009.ix[((lcm2009['default_flag'] == 0) & (lcm2009['dura_fs_attst'] > 15)) | ((lcm2009['default_flag'] == 1) & ((lcm2009['dura_attst_df'] <= 0) | (lcm2009['dura_fs_df'] > 24))), 'pw'] = 9

lcm2009['mkey'] = lcm2009.ix[:, u'entityuen'].map(str) + lcm2009.ix[:, u'pw_year'].map(str)
	
lcm2009_sort = lcm2009.sort(['mkey', 'default_flag', 'pw', 'dura_fybegin_attst_abs'], ascending = [True, False, True, True])
lcm2009_after_pw = lcm2009_sort.ix[((lcm2009_sort['pw'] != 0) & (lcm2009_sort['pw'] != 9)), :].drop_duplicates(['mkey']) 

lcm2009_after_pw['incomp'] = np.where(lcm2009_after_pw.entityuen.map(int).isin(comp.entityuen), 0, 1) 
lcm2009_after_pw.query('incomp == 1').ix[:, ['entityuen', 'final_form_date', 'financial_statement_date', 'dura_fs_attst', 'default_flag', 'pw']]
lcm2009_after_pw.query('incomp == 1').to_excel(r'R:\Global_Market_Risk\RCST\WCM\Huiming_Song\LCM\diff.xlsx', index = False)
 


 ###################################################################################################################################################

lcm2009_after_pw.final_form_date.min()
 
lcm2009_after_pw.final_form_date.max()
 
lcm2009_after_pw.default_date.min()
 
lcm2009_after_pw.default_date.max()
 
lcm2009_after_pw.default_flag.sum()
 
lcm2009_after_pw.shape
 
lcm2009_after_pw.entityuen.nunique()
 
