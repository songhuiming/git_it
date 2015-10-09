
# using new bcis data with source prepared by Echo

import os
import re
import datetime

import numpy as np
import pandas as pd

# echo13['rr'] = map(lambda x: int(re.sub(pattern1, '', x)), echo13.risk_trend)
# echo13['rr2'] = echo13.risk_trend.replace(pattern1, '')
# echo13['eno1'] = map(lambda x: str(x)[0], echo13.entity_no)
# echo13['eno5'] = map(lambda x: str(x)[:5], echo13.entity_no)
# b2013 = echo13.query('type != "ALLOW" & rr <= 80 & eno1 != "N" & eno5 != "00000" & entity_no != "0" ')   #


pattern1 = re.compile(r'\D')

def get_first_2(x):
	return int(x.replace(pattern1, ''))


def read_echo_bcis(year):
	'''
	type != 'ALLOW'
	rr0 <= 80
	entity_no not start with N, or 00000
	'''
	csvfile = r'R:\Global_Market_Risk\RCST\WCM\Huiming_Song\BCIS\BCIS2\EchoResults' + str(year) + '.csv'
	tp = pd.read_csv(csvfile, iterator = True, chunksize = 10000)   # for 2013, shape = (4228354, 29)
	fp = pd.concat(tp, ignore_index = True)
	# fp = pd.read_csv(csvfile, nrows = 100000)   # for test purpose only to read in only 100k rows data 
	print fp.shape
	fp.columns = [x.lower() for x in fp.columns]
	famt = fp.ix[:, ['entity_no', 'amount']].groupby('entity_no').amount.sum().reset_index()
	famt.columns = ['entity_no', 'total_amount']
	fp = pd.merge(left = fp, right = famt, on = 'entity_no', how = 'left')
	fp['rr'] = map(lambda x: int(re.sub(pattern1, '', x)), fp.risk_trend)
	fp['eno1'] = map(lambda x: str(x)[0], fp.entity_no)
	fp['eno5'] = map(lambda x: str(x)[:5], fp.entity_no)
	fp['years'] = map(lambda x: int(x[:4]), fp.tbl_dt)
	fp['months'] = map(lambda x: int(x[5:7]), fp.tbl_dt)
	fp['default_flag'] = np.where((fp.type == 'IMP LOANS') | ( fp.type == 'IMP ADJ'), 1, 0)

	fp_clean_good = fp.query('type != "ALLOW" & rr <= 80 & eno1 != "N" & eno5 != "00000" & entity_no != "0" & default_flag == 0')
	for j in fp_clean_good.tbl_dt.unique():
		month_data_good = fp_clean_good.ix[fp_clean_good.tbl_dt == j, :]
		month_data_good.to_csv(r'R:\Global_Market_Risk\RCST\WCM\Huiming_Song\BCIS\BCIS2\clean_data\bcis_good_' + j.replace('-', '') + '.csv', index = False)
	
	fp_clean_bad = fp.query('type != "ALLOW" & rr <= 80 & eno1 != "N" & eno5 != "00000" & entity_no != "0" & default_flag == 1')
	fp_clean_bad['default_years'] = fp_clean_bad.years
	fp_clean_bad['default_months'] = fp_clean_bad.months
	fp_clean_bad.to_csv(r'R:\Global_Market_Risk\RCST\WCM\Huiming_Song\BCIS\BCIS2\clean_data\bcis_bad_' + str(year) + '.csv', index = False)	
		
	print str(year) + 'done'	


# read in the data from 2005 to 2013. 03/04/05 will be read separately.
for i in range(2009, 2014):
	read_echo_bcis(i)