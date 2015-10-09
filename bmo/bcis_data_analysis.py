import os
import re
import datetime

import numpy as np
import pandas as pd

# tp = pd.read_csv(r'R:\Global_Market_Risk\RCST\WCM\Huiming_Song\BCIS\BCIS2\\EchoResults2013.csv', iterator = True, chunksize = 10000)
# echo13 = pd.concat(tp, ignore_index = True)
# echo05.to_pickle(r'R:\Global_Market_Risk\RCST\WCM\Huiming_Song\BCIS\BCIS2\echo05')


#os.path.basename(r'R:\Global_Market_Risk\RCST\WCM\Huiming_Song\BCIS\good_20021231.csv')
#Out[16]: 'good_20021231.csv'

bcispath = r'R:\Global_Market_Risk\RCST\WCM\Huiming_Song\BCIS\\'
msRating = ['I-1', 'I-2', 'I-3', 'I-4', 'I-5', 'I-6', 'I-7', 'S-1', 'S-2', 'S-3', 'S-4', 'P-1', 'P-2', 'P-3', 'D-1']
ranking = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

bad = pd.read_excel(r'R:\Global_Market_Risk\RCST\WCM\Huiming_Song\BCIS\bcis_bad_first_time.xlsx', sheetname = 'bcis_bad_first_time')
badnames = bad.columns
bad.columns = [x.lower() for x in badnames]
# sum(bad.groupby('entity_no').amount.sum() > 100000)
bad_tot_amt = bad.groupby('entity_no').amount.sum().reset_index()
bad_tot_amt.columns = ['entity_no', 'total_amount']
bad = pd.merge(left = bad, right = bad_tot_amt, on = 'entity_no',how = "left")
bad.sort(['entity_no', 'data_period'], ascending = [True, True], inplace = True)
# bad_entity = bad.drop_duplicates('entity_no').ix[:, ['entity_no', 'default_years', 'default_months']]
#  = bad.query('amount > 100000 & ric != "OTH_NAV"').drop_duplicates('entity_no').ix[:, ['entity_no', 'default_years', 'default_months']]

# bad_entity['default_year'] = [int(re.findall(re.compile(r'(\d{4})\d+'), str(i))[0]) for i in bad_entity.data_period]   #default_years already exist

# read in bad list from CCDM data
# def read_ccdm_def():
	# pcus_bad = pd.read_excel(r'R:\Global_Market_Risk\RCST\WCM\Huiming_Song\BCIS\bmo_defaults.xlsx', sheetname = 'ComUS')
	# pcca_bad = pd.read_excel(r'R:\Global_Market_Risk\RCST\WCM\Huiming_Song\BCIS\bmo_defaults.xlsx', sheetname = 'ComCanada')
	# cm_bad = pd.read_excel(r'R:\Global_Market_Risk\RCST\WCM\Huiming_Song\BCIS\bmo_defaults.xlsx', sheetname = 'Corporate')
	# pcus_bad.columns = [x.lower().replace(' ', '_') for x in pcus_bad.columns]
	# pcca_bad.columns = [x.lower().replace(' ', '_') for x in pcca_bad.columns]
	# cm_bad.columns = [x.lower().replace(' ', '_') for x in cm_bad.columns]
	# pcus_bad['default_years'] = [x.year for x in pcus_bad.default_date]
	# pcus_bad['default_months'] = [x.month for x in pcus_bad.default_date]
	# pcca_bad['default_years'] = [x.year for x in pcca_bad.default_date]
	# pcca_bad['default_months'] = [x.month for x in pcca_bad.default_date]
	# cm_bad['default_years'] = [x.year for x in cm_bad.default_date]
	# cm_bad['default_months'] = [x.month for x in cm_bad.default_date]
	# ccdm_def = pd.concat([pcus_bad, pcca_bad, cm_bad], ignore_index = True)
	# return ccdm_def

# ccdm_def = read_ccdm_def()
# bad_entity = bad.drop_duplicates('entity_no').ix[:, ['entity_no', 'default_years', 'default_months']]
# bad_entity_100k = bad.query('amount > 100000 & ric != "OTH_NAV"').drop_duplicates('entity_no').ix[:, ['entity_no', 'default_years', 'default_months']]
# bad_entity_250k = bad.query('amount > 250000 & ric != "OTH_NAV"').drop_duplicates('entity_no').ix[:, ['entity_no', 'default_years', 'default_months']]
	
bad1_counts = ccdm_def.groupby(['default_years', 'default_months']).size()
bad2_count = bad_entity_100k.groupby(['default_years', 'default_months']).size()
bad3_count = bad_entity_250k.groupby(['default_years', 'default_months']).size()
bad_count_comp = pd.concat([bad1_counts, bad2_count, bad3_count], axis = 1)
bad_count_comp.columns = ['ccdm', 'bcis_100k', 'bcis_250k']
bad_count_comp.reset_index(inplace = True)
bad_count_comp.to_excel(r'R:\Global_Market_Risk\RCST\WCM\Huiming_Song\BCIS\default_compare.xlsx', index = False)

fig = plt.figure(figsize = (16, 12))
ax = fig.add_subplot(1, 1, 1)
rec1 = ax.plot(bad_count_comp.query('bcis_100k < 200').ix[:, 'ccdm'])
rec2 = ax.plot(bad_count_comp.query('bcis_100k < 200').ix[:, 'bcis_100k'])
rec3 = ax.plot(bad_count_comp.query('bcis_100k < 200').ix[:, 'bcis_250k'])
# rec1 = ax.plot(bad_count_comp.query('bcis_100k < 500').ix[:, 'ccdm'])
# rec2 = ax.plot(bad_count_comp.query('bcis_100k < 500').ix[:, 'bcis_100k'])
# rec3 = ax.plot(bad_count_comp.query('bcis_100k < 500').ix[:, 'bcis_250k'])
ax.set_xticks(range(1, bad_count_comp.shape[0], 12))
ax.set_xticklabels(bad_count_comp.default_years.unique())
ax.legend( (rec1[0], rec2[0], rec3[0]), ('ccdm', 'bcis>100k', 'bcis>250k'), loc = 0)
plt.show()

bad_entity_250k = bad.query('total_amount > 250000 & ric != "OTH_NAV"').drop_duplicates('entity_no').ix[:, ['entity_no', 'default_years', 'default_months']]

def data_prep():
	final_data = pd.DataFrame()    #good snapshots
	for i in os.listdir(bcispath):
		# select each year end data
		if 'good' in i and '1231' in i:
		#if i == 'good_20101231.csv':
			print i
			snapshot = pd.read_csv(bcispath + i)
			snapshot.columns = [x.lower() for x in snapshot.columns]
			snapshot_tot_amt = snapshot.groupby('entity_no').amount.sum().reset_index()
			snapshot_tot_amt.columns = ['entity_no', 'total_amount']
			snapshot = pd.merge(left = snapshot, right = snapshot_tot_amt, on = 'entity_no',how = "left")
			snapshot['risk_ranking'] = snapshot.ix[:, 'msr'].map(dict(zip(msRating, ranking)))
			snapshot['years'] = [int(re.findall(re.compile(r'(\d{4})\d+'), str(i))[0]) for i in snapshot.data_period]
			snapshot.sort(['entity_no', 'risk_ranking'], ascending = [True, False], inplace = True)  		# sort in data 
			snapshot_entity = snapshot.query('total_amount > 250000 & ric != "OTH_NAV"').drop_duplicates('entity_no')
			good_bad = pd.merge(left = snapshot_entity, right = bad_entity_250k, on = 'entity_no', how = 'left')
			good_bad['yr_diff'] = good_bad.default_years - good_bad.years
			snapshot_w_bad = good_bad.ix[(good_bad.yr_diff > 0) | (good_bad.yr_diff.isnull()), :]
			final_data = pd.concat([final_data, snapshot_w_bad], axis = 0, ignore_index = True)
	return final_data

final_data = data_prep()     # shape = (770452, 32)
	
def sum_data(segment, rr):
	# select data by segment and risk_ranking
	f = final_data.ix[(final_data.sub_group == segment) & (final_data.risk_ranking == rr), :]
	default_num = f.groupby(['years', 'risk_ranking', 'yr_diff']).size().unstack()
	tot_num = f.groupby(['years', 'risk_ranking']).size() 
	summary_info = pd.concat([tot_num, default_num], axis = 1)
	# rename the counts 
	summary_info.columns = ['total_obgs'] + ['yr_' + str(i) for i in range(1, summary_info.shape[1])]
	# calculate the ratios
	for j in range(1, summary_info.shape[1]):
		# summary_info['diff_' + str(j)] = summary_info.iloc[:, j] / summary_info.iloc[:, 0]
		summary_info['cum_yr_' + str(j)] = np.sum(summary_info.iloc[:, 1: (j + 1)], axis = 1) / summary_info.iloc[:, 0]
	# to make the output is alike up-triangle  	
	counts_col = [x for x in summary_info.columns if x.startswith('yr_')]
	cums_col = [x for x in summary_info.columns if x.startswith('cum_yr_')]
	counts = summary_info.ix[:, counts_col]
	cums = summary_info.ix[:, cums_col]
	for i in range(counts.shape[0]):
		for j in range(counts.shape[1]):
			if np.isnan(counts.iloc[i, j]):
				cums.iloc[i, j] = np.nan		
	summary_info.ix[:, cums_col] = cums
	# outfile = r'R:\Global_Market_Risk\RCST\WCM\Huiming_Song\BCIS\summary_info_' + segment + '_' + str(rr) + '_' + datetime.date.today().strftime("%Y%m%d") + '.xlsx'	
	output = summary_info.query('2002 <= years <= 2012')
	# output.reset_index().to_excel(outfile, index = False)	
	return output


writer = pd.ExcelWriter(u"R:\Global_Market_Risk\RCST\WCM\Huiming_Song\BCIS\summary_info_PCUS_" + datetime.date.today().strftime("%Y%m%d") + '.xlsx') 

for i in range(8, 9):
	pcus = sum_data('PC_US', i)	
	pcus.to_excel(writer, sheet_name = 'rr_' + str(i))
	print pcus.iloc[:, 10:].to_string()

writer.save()