## this is from ifrs9_pcus_cni_I2to7.py

# part 1 is to build model
# part 2 is to predict for given starting year+quarter

import pandas as pd
import numpy as np
import itertools
import pickle
from itertools import chain, combinations
import statsmodels.formula.api as smf
import scipy.stats as stats
from scipy.stats import norm
import statsmodels.api as sm
import matplotlib.pyplot as plt
%matplotlib inline 
plt.rcParams['figure.figsize'] = (16, 12)



 

def save_obj(obj, name):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
		
######################################################  Part I: Build model  #######################################################  
def us_ci_pd(ratings):

	# final_return = pd.DataFrame(columns = ['start_rating', 'pred_model', 'mean_stdev', 'model_raw_data', 'model_clean_data', 'model_fit_result'])
	final_return = {}
	
	cm = pd.read_excel('H:\work\IFRS9\PCUS\CI\pc_us_cnt_cum_pd_from_sas.xlsx', sheetname = ratings)   # cum pd from sas, need to normalize it
	cumcol = [x for x in cm.columns if x.startswith('cumpd')]
	print cm.ix[:, cumcol].shape
	cm_cum = cm.ix[:, cumcol]
	nc = cm.ix[:, cumcol].shape[1]
	nl = cm.ix[:, cumcol].shape[0]

	mean_stdev = pd.DataFrame(np.array([np.nanmean(cm_cum, axis = 0), np.nanstd(cm_cum, axis = 0)]).T, columns = ['d_mean', 'd_stdev'])
	
	# normalize cm_cum
	def normalize_data(indata, mean_stdev = mean_stdev):
		output = pd.DataFrame(columns = indata.columns)
		for i in range(43):
			try:
				output.iloc[:, i] = (indata.ix[:, i] - mean_stdev.ix[i, 0]) / mean_stdev.ix[i, 1]
			except:
				pass
		return output

	cm_cum_norm = normalize_data(cm_cum)


	#左上三角矩阵按行拉直 			# 从cumpd1 到cumpd25，总共25列，len=775
	def tri_to_line(indata):
		# indata是左上三角阵，outdata是一个向量或list
		outdata = []   
		for i in range(43):
			if i <= 18:
				for j in range(25):
					outdata.append(indata.iloc[i, j])
			else:
				for j in range(43 - i):
					outdata.append(indata.iloc[i, j])
		return outdata

	cumpd_incol = tri_to_line(cm_cum_norm)    

	# read in historical GDP info
	gdp = pd.read_excel(r'H:\work\IFRS9\macro_economic_base.xlsx', sheetname = 'gdp_test')
	gdp['year'] = [int('20'+x.split(':')[0].strip()) for x in gdp.yyyq]
	gdp['qtr'] = [int(x.split(':')[1].strip()) for x in gdp.yyyq]
	gdp['yq'] = [str('20'+x.split(':')[0].strip()) + '0' + str(x.split(':')[1].strip()) for x in gdp.yyyq]


	# set up point in time
	yq_p0 = gdp.ix[5:, 'yq']     # from 2003Q2 to 2013Q4
	yq_p1 = gdp.ix[4:46, 'yq']
	yq_p2 = gdp.ix[3:45, 'yq']
	yq_p3 = gdp.ix[2:44, 'yq']

	yq_pit0_incol = [str(x) for ll in [yq_p0[i:(i + 25)] for i in range(43)] for x in ll] 		# len=775
	yq_pit1_incol = [str(x) for ll in [yq_p1[i:(i + 25)] for i in range(43)] for x in ll]
	yq_pit2_incol = [str(x) for ll in [yq_p2[i:(i + 25)] for i in range(43)] for x in ll]
	yq_pit3_incol = [str(x) for ll in [yq_p3[i:(i + 25)] for i in range(43)] for x in ll]

	# set up starting time
	yq_v0 = gdp.ix[4:46, 'yq']     	#from 2003Q1 to 2013Q3
	yq_v1 = gdp.ix[3:45, 'yq']		#from 2002Q4 to 2013Q2
	yq_v2 = gdp.ix[2:44, 'yq']
	yq_v3 = gdp.ix[1:43, 'yq']

	def starting_data(indata):
		outdata = []
		for i in range(43):
			if i <= 18:
				for j in range(25):
					outdata.append(indata.iloc[i])
			else:
				for j in range(43 - i):
					outdata.append(indata.iloc[i])
		return outdata

	yq_v0_incol = starting_data(yq_v0)   
	yq_v1_incol = starting_data(yq_v1)
	yq_v2_incol = starting_data(yq_v2)
	yq_v3_incol = starting_data(yq_v3)

	df = pd.DataFrame([yq_v0_incol, yq_v1_incol, yq_v2_incol, yq_v3_incol, yq_pit0_incol, yq_pit1_incol, yq_pit2_incol, yq_pit3_incol, cumpd_incol]).T
	df.columns = ['yq_v0_incol', 'yq_v1_incol', 'yq_v2_incol', 'yq_v3_incol', 'yq_pit0_incol', 'yq_pit1_incol', 'yq_pit2_incol', 'yq_pit3_incol', 'cum_pd']

	# df['inverse_cpd'] = (df.cum_pd + 10E-6).map(lambda x: norm.ppf(x))
	df['cum_pd_num'] = [float(x) for x in df.ix[:, 'cum_pd']]

	# join data to get macro economics info
	#1 gdp
	df['gdp_v1'] = df.yq_v1_incol.map(dict(zip(gdp['yq'], gdp['gdp'])))
	df['gdp_v2'] = df.yq_v2_incol.map(dict(zip(gdp['yq'], gdp['gdp'])))
	df['gdp_v3'] = df.yq_v3_incol.map(dict(zip(gdp['yq'], gdp['gdp'])))

	df['gdp_p1'] = df.yq_pit1_incol.map(dict(zip(gdp['yq'], gdp['gdp'])))
	df['gdp_p2'] = df.yq_pit2_incol.map(dict(zip(gdp['yq'], gdp['gdp'])))
	df['gdp_p3'] = df.yq_pit3_incol.map(dict(zip(gdp['yq'], gdp['gdp'])))
	#2 bbb bond yield rate
	df['bbbspread_v1'] = df.yq_v1_incol.map(dict(zip(gdp['yq'], gdp['bbbspread'])))
	df['bbbspread_v2'] = df.yq_v2_incol.map(dict(zip(gdp['yq'], gdp['bbbspread'])))
	df['bbbspread_v3'] = df.yq_v3_incol.map(dict(zip(gdp['yq'], gdp['bbbspread'])))

	df['bbbspread_p1'] = df.yq_pit1_incol.map(dict(zip(gdp['yq'], gdp['bbbspread'])))
	df['bbbspread_p2'] = df.yq_pit2_incol.map(dict(zip(gdp['yq'], gdp['bbbspread'])))
	df['bbbspread_p3'] = df.yq_pit3_incol.map(dict(zip(gdp['yq'], gdp['bbbspread'])))

	# quarters passed
	def quarters():
		outdata = []
		for i in range(43):
			if i <= 18:
				for j in range(25):
					outdata.append(j)
			else:
				for j in range(43 - i):
					outdata.append(j)
		return outdata

	df['quarters_'] = quarters()

	f = 'cum_pd_num~gdp_v1+gdp_v2+gdp_v3+gdp_p1+gdp_p2+gdp_p3+bbbspread_v1+bbbspread_v2+bbbspread_v3+bbbspread_p1+bbbspread_p2+bbbspread_p3'

	## loop through combinations
	res_df = pd.DataFrame(columns = ['f', 'num_of_x', 'adj_rsquare'])
	for i in ['gdp_v1', 'gdp_v2', 'gdp_v3', '']:
		for j in ['gdp_p1', 'gdp_p2', 'gdp_p3', '']:
			for k in ['bbbspread_v1', 'bbbspread_v2', 'bbbspread_v3', '']:
				for s in ['bbbspread_p1', 'bbbspread_p2', 'bbbspread_p3', '']:
					try:
						y = df.cum_pd_num
						X = df.ix[:, [i, j, k, s, 'quarters_']].dropna(axis = 1, how = 'all')
						lm = sm.OLS(y, X).fit()
						score = lm.rsquared_adj
						vars = X.columns
						#print score
						res = pd.DataFrame(['cum_pd_num~' + '+'.join(vars), len(vars), score]).T
						res.columns = ['f', 'num_of_x', 'adj_rsquare']
						#print res
						res_df = res_df.append(res)
					except:
						pass

	res_df.sort('adj_rsquare', ascending = False, inplace = True)
	res_df.set_index(np.arange(res_df.shape[0]), inplace = True)					

	 
	f = res_df.ix[0, 'f']
	lm = smf.ols(formula = str(f), data = df).fit()
	print lm.summary()


	# from column to upper triangle matirx: 775 to cm_cum shape
	ypred = lm.predict()
	def line_to_tri(indata):
		# indata是左上三角阵，outdata是一个向量或list
		# 前面19行每行25个值，从20行开始每行少一个值
		outdata = pd.DataFrame(columns = cm_cum.columns, index = cm_cum.index)  
		for i in range(43):
			if i <= 18:
				for j in range(25):
					outdata.iloc[i, j] = indata[i * 25 + j]
			else:
				already = sum([43 - x for x in range(19, i)])
				for j in range(43 - i):
					outdata.iloc[i, j] = indata[19 * 25 + already + j]
		return outdata

	ypred_to_tri = line_to_tri(ypred.tolist())

	 
	for i in range(25):
		ypred_to_tri.ix[:, i] = ypred_to_tri.ix[:, i] * mean_stdev.ix[i, 'd_stdev'] + mean_stdev.ix[i, 'd_mean']

	# plot year 1, year 3, year 5 actual value v.s. predicted value	
 
	# year1
	fig = plt.figure(figsize = (16, 12))
	ax = fig.add_subplot(111)
	ax.plot(range(1, 41), ypred_to_tri.ix[:39, 'cumpd4'], color = 'r', linestyle = '-', linewidth = 2, markeredgecolor='none', marker = '', label = r'pred')
	ax.plot(range(1, 41), cm_cum.ix[:39, 'cumpd4'], color = 'b', linestyle = '-', linewidth = 2, markeredgecolor='none', marker = '', label = r'true')
	ax.axhline(y = cm_cum.ix[:39, 'cumpd4'].mean())
	ax.annotate('average of actual PD', (5, cm_cum.ix[:39, 'cumpd4'].mean()), xytext = (0.2, 0.5), fontsize = 20, textcoords = 'axes fraction', arrowprops = dict(facecolor = 'grey', color = 'grey'))
	plt.title('Year 1', fontsize = 20)
	plt.legend(loc = 3, ncol = 1)  

	# year 3
	fig = plt.figure(figsize = (16, 12))
	ax = fig.add_subplot(111)
	ax.plot(range(1, 33), ypred_to_tri.ix[:31, 'cumpd12'], color = 'r', linestyle = '-', linewidth = 2, markeredgecolor='none', marker = '', label = r'pred')
	ax.plot(range(1, 33), cm_cum.ix[:31, 'cumpd12'], color = 'b', linestyle = '-', linewidth = 2, markeredgecolor='none', marker = '', label = r'true')
	ax.axhline(y = cm_cum.ix[:31, 'cumpd12'].mean())
	ax.annotate('average of actual PD', (20, cm_cum.ix[:31, 'cumpd12'].mean()), xytext = (0.7, 0.2), fontsize = 20, textcoords = 'axes fraction', arrowprops = dict(facecolor = 'grey', color = 'grey'))
	plt.title('Year 3', fontsize = 20)
	plt.legend(loc = 3, ncol = 1) 
	 
	# year 5 
	fig = plt.figure(figsize = (16, 12))
	ax = fig.add_subplot(111)
	ax.plot(range(1, 25), ypred_to_tri.ix[:23, 'cumpd20'], color = 'r', linestyle = '-', linewidth = 2, markeredgecolor='none', marker = '', label = r'pred')
	ax.plot(range(1, 25), cm_cum.ix[:23, 'cumpd20'], color = 'b', linestyle = '-', linewidth = 2, markeredgecolor='none', marker = '', label = r'true')
	ax.axhline(y = cm_cum.ix[:23, 'cumpd20'].mean())
	ax.annotate('average of actual PD', (20, cm_cum.ix[:23, 'cumpd20'].mean()), xytext = (0.7, 0.2), fontsize = 20, textcoords = 'axes fraction', arrowprops = dict(facecolor = 'grey', color = 'grey'))
	plt.title('Year 5', fontsize = 20)
	plt.legend(loc = 3, ncol = 1) 	
		
	# ['start_rating', 'pred_model', 'mean_stdev', 'model_raw_data', 'model_clean_data', 'model_fit_result']
	final_return['start_rating'] = ratings
	final_return['pred_model'] = lm
	final_return['mean_stdev'] = mean_stdev
	final_return['model_raw_data'] = cm
	final_return['model_clean_data'] = df
	final_return['model_fit_result'] = ypred_to_tri
	return final_return

############### run model for each rating and get the output 
us_ci_model = {}	
us_ci_model['I2to7'] = us_ci_pd('I2to7')
us_ci_model['S1'] = us_ci_pd('S1')
us_ci_model['S2'] = us_ci_pd('S2')
us_ci_model['S34'] = us_ci_pd('S34')
us_ci_model['P123'] = us_ci_pd('P123')

# pickle.dump(us_ci_model, open(r'H:\work\IFRS9\PCUS\CI\us_ci_model.pkl', 'wb'))
# us_ci_readin_model = pickle.load(open(r'H:\work\IFRS9\PCUS\CI\us_ci_model.pkl', 'rb'))

############### end to run model for each rating and get the output 




 

#############################################  Part II: Cumulative PD predict for given start time   #########################################################  

# to predict for a give time,   be careful with   "200803"   to replace
# 大概： 对我们要predict的那个year qtr， 比如 ‘200803’， 找出它的 v1 v2 v3 时间 , 分别为(‘200802’ ‘200801’ ‘200704’)

yr_qtr = pd.read_excel(r'H:\work\IFRS9\PCUS\CI\200803_current_pit_macroeconomics.xlsx', sheetname = 'yr_qtr')
macro_hist = pd.read_excel(r'H:\work\IFRS9\PCUS\CI\200803_current_pit_macroeconomics.xlsx', sheetname = 'macro_hist') 
yr_qtr['yr_qtr_str'] = [str(x) for x in yr_qtr.yr_qtr_num]
macro_hist['yr_qtr_str'] = [str(x) for x in macro_hist.yr_qtr_num]

def us_ci_pred(year_qtr, rating, readin_model = us_ci_model):
	current_yq_index = yr_qtr.ix[yr_qtr.yr_qtr_str == year_qtr, 'num_index'].values[0]    # '200803' will be 35
	v1_yq_index = current_yq_index - 1
	v2_yq_index = current_yq_index - 2
	v3_yq_index = current_yq_index - 3
	p1_yq_index = current_yq_index - 0
	p2_yq_index = current_yq_index - 1
	p3_yq_index = current_yq_index - 2

	pred_df = pd.DataFrame()

	pred_df['yq_v0_incol'] = [yr_qtr.ix[yr_qtr.num_index == current_yq_index, 'yr_qtr_str'].values[0]]  * (max(yr_qtr.num_index) - current_yq_index + 1)
	pred_df['yq_v1_incol'] = [yr_qtr.ix[yr_qtr.num_index == v1_yq_index, 'yr_qtr_str'].values[0]]  * (max(yr_qtr.num_index) - current_yq_index + 1)
	pred_df['yq_v2_incol'] = [yr_qtr.ix[yr_qtr.num_index == v2_yq_index, 'yr_qtr_str'].values[0]]  * (max(yr_qtr.num_index) - current_yq_index + 1)
	pred_df['yq_v3_incol'] = [yr_qtr.ix[yr_qtr.num_index == v3_yq_index, 'yr_qtr_str'].values[0]]  * (max(yr_qtr.num_index) - current_yq_index + 1)

	pred_df['yq_pit1_incol'] = yr_qtr.ix[(p1_yq_index - 1) : (max(yr_qtr.num_index) - 1), 'yr_qtr_str'].values
	pred_df['yq_pit2_incol'] = yr_qtr.ix[(p2_yq_index - 1) : (max(yr_qtr.num_index) - 2), 'yr_qtr_str'].values
	pred_df['yq_pit3_incol'] = yr_qtr.ix[(p3_yq_index - 1) : (max(yr_qtr.num_index) - 3), 'yr_qtr_str'].values
	 
	pred_df['gdp_v1'] = pred_df['yq_v1_incol'].map(dict(zip(macro_hist['yr_qtr_str'], macro_hist['norminal_gdp'])))
	pred_df['gdp_v2'] = pred_df['yq_v2_incol'].map(dict(zip(macro_hist['yr_qtr_str'], macro_hist['norminal_gdp'])))
	pred_df['gdp_v3'] = pred_df['yq_v3_incol'].map(dict(zip(macro_hist['yr_qtr_str'], macro_hist['norminal_gdp'])))
	pred_df['gdp_p1'] = pred_df['yq_pit1_incol'].map(dict(zip(macro_hist['yr_qtr_str'], macro_hist['norminal_gdp'])))
	pred_df['gdp_p2'] = pred_df['yq_pit2_incol'].map(dict(zip(macro_hist['yr_qtr_str'], macro_hist['norminal_gdp'])))
	pred_df['gdp_p3'] = pred_df['yq_pit3_incol'].map(dict(zip(macro_hist['yr_qtr_str'], macro_hist['norminal_gdp'])))
	pred_df['bbbspread_v1'] = pred_df['yq_v1_incol'].map(dict(zip(macro_hist['yr_qtr_str'], macro_hist['bbbspread'])))
	pred_df['bbbspread_v2'] = pred_df['yq_v2_incol'].map(dict(zip(macro_hist['yr_qtr_str'], macro_hist['bbbspread'])))
	pred_df['bbbspread_v3'] = pred_df['yq_v3_incol'].map(dict(zip(macro_hist['yr_qtr_str'], macro_hist['bbbspread'])))
	pred_df['bbbspread_p1'] = pred_df['yq_pit1_incol'].map(dict(zip(macro_hist['yr_qtr_str'], macro_hist['bbbspread'])))
	pred_df['bbbspread_p2'] = pred_df['yq_pit2_incol'].map(dict(zip(macro_hist['yr_qtr_str'], macro_hist['bbbspread'])))
	pred_df['bbbspread_p3'] = pred_df['yq_pit3_incol'].map(dict(zip(macro_hist['yr_qtr_str'], macro_hist['bbbspread']))) 
	pred_df['quarters_'] = np.arange((max(yr_qtr.num_index) - current_yq_index + 1))

	pred_df.dropna(axis = 0, how = 'any')
	# lm.predict(pred_df.dropna(axis = 0, how = 'any'))

	## pick 25 quarters
	mean_stdev = readin_model[rating]['mean_stdev']
	return readin_model[rating]['pred_model'].predict(pred_df.dropna(axis = 0, how = 'any'))[:25] * mean_stdev.ix[:24, 'd_stdev'] + mean_stdev.ix[:24, 'd_mean']

us_ci_pred('200803', 'I2to7', us_ci_model)

# compare pred vs acturl for each rating, and next 25 quarters for each starting quarter
## macro economics can only pred to 201202, which is 37th in the yq
### we can remove this part in the final calculation
def compare_pred_actual():
	tttt = [str(x) for x in pd.read_excel('H:\work\IFRS9\PCUS\CI\pc_us_cnt_cum_pd_from_sas.xlsx', sheetname = 'S1').yq]	
	df_out = pd.DataFrame(columns = ['pred_from_model', 'actual_data', 'pred_from_fitting', 'rating', 'yq'])
	for j in ['I2to7', 'S1', 'S2', 'S34', 'P123']:	
		for i in range(38):  #macro economics can only pred to 201202, which is 37th in the yq
			s1 = us_ci_pred(tttt[i], j, us_ci_model)   # pred 25 quarters
			s2 = us_ci_model[j]['model_raw_data'].ix[i, 3:28].values     # true 25 quarters
			s3 = us_ci_model[j]['model_fit_result'].ix[i, :25].values     # pred from model fitting time
			data1 = pd.DataFrame([s1, s2, s3]).T
			data1.columns = ['pred_from_model', 'actual_data', 'pred_from_fitting'] 
			data1['rating'] = [j] * 25
			data1['yq'] = [tttt[i]] * 25
			df_out = pd.concat([df_out, data1], axis = 0, ignore_index = True)
	return df_out

comp_pred_actual = compare_pred_actual()
# comp_pred_actual.to_excel('H:\work\IFRS9\PCUS\CI\us_ci_pred_vs_actual.xlsx')

	
## prepare the default curve table for each rating

def default_curve(yq, rating):
	cum_pd = pd.DataFrame(us_ci_pred(yq, rating, us_ci_model)) 
	cum_pd.columns = ['cumulative_pd']
	cum_pd['month_i'] = range(3, (cum_pd.shape[0] + 1)* 3, 3)

	# add 0 for first row
	row0 = pd.DataFrame([0, 0]).T
	row0.columns = cum_pd.columns
	cum_pd = pd.concat([row0, cum_pd], axis = 0)

	cum_pd['month_incr'] = cum_pd.cumulative_pd.diff().fillna(0) / 3

	# from 78 to 360
	cum_after = pd.DataFrame(columns = cum_pd.columns)
	cum_after['month_i'] = np.arange(cum_pd['month_i'].max() + 3, 360 + 3, 3)		# need to change 360?
	cum_after['month_incr'] = cum_pd.month_incr.values[-1]

	cum_after.ix[0, 'cumulative_pd'] = cum_pd.cumulative_pd.values[-1] * 2 - cum_pd.cumulative_pd.values[-2] 
	for i in range(1, cum_after.shape[0]):
		cum_after.ix[i, 'cumulative_pd'] = cum_after.ix[i - 1, 'cumulative_pd'] + cum_pd.cumulative_pd.values[-1] - cum_pd.cumulative_pd.values[-2] 

	final_cum_pd = pd.concat([cum_pd, cum_after], ignore_index = True)

	# part 2: create default_rate curve
	def_rate_curve = pd.DataFrame(columns = ['month_i', 'rate'])
	def_rate_curve.month_i = np.arange(1, 360 + 1)		# need to change 360?
	def_rate_curve.rate = final_cum_pd.ix[np.digitize(def_rate_curve.month_i, final_cum_pd.month_i) - 1, 'month_incr'].values
	return def_rate_curve

default_curve_data = {}	
default_curve_data['I2to7'] = default_curve('200803', 'I2to7')
default_curve_data['S1'] = default_curve('200803', 'S1')
default_curve_data['S2'] = default_curve('200803', 'S2')
default_curve_data['S34'] = default_curve('200803', 'S34')
default_curve_data['P123'] = default_curve('200803', 'P123')
