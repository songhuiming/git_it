## this is from ifrs9_pcus_cni_I2to7.py
## v2: update on 20151007, update the data prep part to get the anchor_point and automatic the model run part

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

	mean_stdev = pd.DataFrame(np.array([np.nanmean(cm_cum, axis = 0), np.nanstd(cm_cum, axis = 0)]).T, columns = ['d_mean', 'd_stdev'])
	anchor_point = np.argmax(mean_stdev.d_mean.diff().fillna(method = 'backfill') < 0)

	# normalize cm_cum
	def normalize_data(indata, mean_stdev = mean_stdev):
		output = pd.DataFrame(columns = indata.columns)
		for i in range(indata.shape[1]):   # 43 列
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
		for i in range(indata.shape[0]):   # 43 行
			if i <= (indata.shape[0] - anchor_point):   # i 是行
				for j in range(anchor_point):    # j 是列
					outdata.append(indata.iloc[i, j])
			else:
				for j in range(indata.shape[0] - i):
					outdata.append(indata.iloc[i, j])
		return outdata

	cumpd_incol = tri_to_line(cm_cum_norm)    

	##########################################   new code to get macro economics   ##################################
	yr_qtr = pd.read_excel(r'H:\work\IFRS9\PCUS\CI\200803_current_pit_macroeconomics.xlsx', sheetname = 'yr_qtr')
	macro_hist = pd.read_excel(r'H:\work\IFRS9\PCUS\CI\200803_current_pit_macroeconomics.xlsx', sheetname = 'macro_hist') 
	yr_qtr['yr_qtr_num'] = [str(x) for x in yr_qtr.yr_qtr_num]
	# macro_hist['yr_qtr_num'] = [str(x) for x in macro_hist.yr_qtr_num]

	traindata_yq_start_index = macro_hist.ix[macro_hist.yr_qtr_num == cm.yq.min(), 'num_index'].values[0]    # start from 196601, 200301 will be 149th (value=148)
	v0_yq_index = traindata_yq_start_index  
	v1_yq_index = traindata_yq_start_index - 1
	v2_yq_index = traindata_yq_start_index - 2
	v3_yq_index = traindata_yq_start_index - 3
	p0_yq_index = traindata_yq_start_index + 1
	p1_yq_index = traindata_yq_start_index - 0
	p2_yq_index = traindata_yq_start_index - 1
	p3_yq_index = traindata_yq_start_index - 2

	# set up point in time
	yq_p0 = macro_hist.ix[p0_yq_index:(p0_yq_index + cm.shape[0] - 1), 'yr_qtr_num']     # from 2003Q2 to 2013Q4,  43 quarters  = the num of row of cm
	yq_p1 = macro_hist.ix[p1_yq_index:(p1_yq_index + cm.shape[0] - 1), 'yr_qtr_num']	 # from 2003Q1 to 2013Q3 	
	yq_p2 = macro_hist.ix[p2_yq_index:(p2_yq_index + cm.shape[0] - 1), 'yr_qtr_num']	 # from 2002Q4 to 2013Q2 
	yq_p3 = macro_hist.ix[p3_yq_index:(p3_yq_index + cm.shape[0] - 1), 'yr_qtr_num']

	yq_pit0_incol = [x for ll in [yq_p0[i:(i + anchor_point)] for i in range(cm.shape[0])] for x in ll] 		# if anchor_point = 25, then len=775
	yq_pit1_incol = [x for ll in [yq_p1[i:(i + anchor_point)] for i in range(cm.shape[0])] for x in ll]
	yq_pit2_incol = [x for ll in [yq_p2[i:(i + anchor_point)] for i in range(cm.shape[0])] for x in ll]
	yq_pit3_incol = [x for ll in [yq_p3[i:(i + anchor_point)] for i in range(cm.shape[0])] for x in ll]

	# set up starting time
	yq_v0 = macro_hist.ix[v0_yq_index:(v0_yq_index + cm.shape[0] - 1), 'yr_qtr_num']     	#from 2003Q1 to 2013Q3, len = 43 
	yq_v1 = macro_hist.ix[v1_yq_index:(v1_yq_index + cm.shape[0] - 1), 'yr_qtr_num'] 		#from 2002Q4 to 2013Q2
	yq_v2 = macro_hist.ix[v2_yq_index:(v2_yq_index + cm.shape[0] - 1), 'yr_qtr_num'] 		#from 200203 to 
	yq_v3 = macro_hist.ix[v3_yq_index:(v3_yq_index + cm.shape[0] - 1), 'yr_qtr_num']        #from 200202 to 201204

	def starting_data(indata):
		outdata = []
		for i in range(cm.shape[0]):
			if i <= (cm.shape[0] - anchor_point):
				for j in range(anchor_point):
					outdata.append(indata.iloc[i])
			else:
				for j in range(cm.shape[0] - i):
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

	for macro_var in macro_hist.columns.values[2:]:
		for lag in range(1, 4):   # lag1 lag2 lag3
			df[macro_var + '_v' + str(lag)] = df.ix[:, 'yq_v' + str(lag) + '_incol'].map(dict(zip(macro_hist['yr_qtr_num'], macro_hist[macro_var])))
			df[macro_var + '_p' + str(lag)] = df.ix[:, 'yq_pit' + str(lag) + '_incol'].map(dict(zip(macro_hist['yr_qtr_num'], macro_hist[macro_var])))
			
	# quarters passed
	def quarters():
		outdata = []
		for i in range(cm_cum.shape[0]):
			if i <= (cm_cum.shape[0] - anchor_point):
				for j in range(anchor_point):
					outdata.append(j)
			else:
				for j in range(cm_cum.shape[0] - i):
					outdata.append(j)
		return outdata

	df['quarters_'] = quarters()

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
		for i in range(cm_cum.shape[0]):   #43行，对应与 cm_cum 行,
			if i <= (cm_cum.shape[0] - anchor_point):    # 对刚开始的长方形区域, 共计cm_cum.shape[0] - anchor_point行， 
				for j in range(anchor_point):			 # anchor_point 列
					outdata.iloc[i, j] = indata[i * anchor_point + j]
			else:
				already = sum([cm_cum.shape[0] - x for x in range((cm_cum.shape[0] - anchor_point) + 1, i)])   # 长方形后面每行会依次用去这么多
				for j in range(cm_cum.shape[0] - i):
					outdata.iloc[i, j] = indata[((cm_cum.shape[0] - anchor_point) + 1) * anchor_point + already + j]              # 后面每行开始减少
		return outdata

	ypred_to_tri = line_to_tri(ypred.tolist())

	 
	for i in range(anchor_point):
		ypred_to_tri.ix[:, i] = ypred_to_tri.ix[:, i] * mean_stdev.ix[i, 'd_stdev'] + mean_stdev.ix[i, 'd_mean']

	# plot year 1, year 3, year 5 actual value v.s. predicted value	
 
	###### year1
	# fig = plt.figure(figsize = (16, 12))
	# ax = fig.add_subplot(111)
	# ax.plot(range(1, 41), ypred_to_tri.ix[:39, 'cumpd4'], color = 'r', linestyle = '-', linewidth = 2, markeredgecolor='none', marker = '', label = r'pred')
	# ax.plot(range(1, 41), cm_cum.ix[:39, 'cumpd4'], color = 'b', linestyle = '-', linewidth = 2, markeredgecolor='none', marker = '', label = r'true')
	# ax.axhline(y = cm_cum.ix[:39, 'cumpd4'].mean())
	# ax.annotate('average of actual PD', (5, cm_cum.ix[:39, 'cumpd4'].mean()), xytext = (0.2, 0.5), fontsize = 20, textcoords = 'axes fraction', arrowprops = dict(facecolor = 'grey', color = 'grey'))
	# plt.title('Year 1', fontsize = 20)
	# plt.legend(loc = 3, ncol = 1)  

	###### year 3
	# fig = plt.figure(figsize = (16, 12))
	# ax = fig.add_subplot(111)
	# ax.plot(range(1, 33), ypred_to_tri.ix[:31, 'cumpd12'], color = 'r', linestyle = '-', linewidth = 2, markeredgecolor='none', marker = '', label = r'pred')
	# ax.plot(range(1, 33), cm_cum.ix[:31, 'cumpd12'], color = 'b', linestyle = '-', linewidth = 2, markeredgecolor='none', marker = '', label = r'true')
	# ax.axhline(y = cm_cum.ix[:31, 'cumpd12'].mean())
	# ax.annotate('average of actual PD', (20, cm_cum.ix[:31, 'cumpd12'].mean()), xytext = (0.7, 0.2), fontsize = 20, textcoords = 'axes fraction', arrowprops = dict(facecolor = 'grey', color = 'grey'))
	# plt.title('Year 3', fontsize = 20)
	# plt.legend(loc = 3, ncol = 1) 
	 
	###### year 5 
	# fig = plt.figure(figsize = (16, 12))
	# ax = fig.add_subplot(111)
	# ax.plot(range(1, 25), ypred_to_tri.ix[:23, 'cumpd20'], color = 'r', linestyle = '-', linewidth = 2, markeredgecolor='none', marker = '', label = r'pred')
	# ax.plot(range(1, 25), cm_cum.ix[:23, 'cumpd20'], color = 'b', linestyle = '-', linewidth = 2, markeredgecolor='none', marker = '', label = r'true')
	# ax.axhline(y = cm_cum.ix[:23, 'cumpd20'].mean())
	# ax.annotate('average of actual PD', (20, cm_cum.ix[:23, 'cumpd20'].mean()), xytext = (0.7, 0.2), fontsize = 20, textcoords = 'axes fraction', arrowprops = dict(facecolor = 'grey', color = 'grey'))
	# plt.title('Year 5', fontsize = 20)
	# plt.legend(loc = 3, ncol = 1) 	
		
	# ['start_rating', 'pred_model', 'mean_stdev', 'model_raw_data', 'model_clean_data', 'model_fit_result']
	final_return['start_rating'] = ratings
	final_return['anchor_point'] = anchor_point
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

# yr_qtr = pd.read_excel(r'H:\work\IFRS9\PCUS\CI\200803_current_pit_macroeconomics.xlsx', sheetname = 'yr_qtr')
macro_hist = pd.read_excel(r'H:\work\IFRS9\PCUS\CI\200803_current_pit_macroeconomics.xlsx', sheetname = 'macro_hist') 
 

def us_ci_pred(year_qtr, rating, readin_model = us_ci_model):
	current_yq_index = macro_hist.ix[macro_hist.yr_qtr_num == year_qtr, 'num_index'].values[0]    # 200803 will be 170
	v1_yq_index = current_yq_index - 1
	v2_yq_index = current_yq_index - 2
	v3_yq_index = current_yq_index - 3
	p1_yq_index = current_yq_index - 0
	p2_yq_index = current_yq_index - 1
	p3_yq_index = current_yq_index - 2

	pred_df = pd.DataFrame()

	pred_df['yq_v0_incol'] = [macro_hist.ix[macro_hist.num_index == current_yq_index, 'yr_qtr_num'].values[0]]  * (max(macro_hist.num_index) - current_yq_index + 1)
	pred_df['yq_v1_incol'] = [macro_hist.ix[macro_hist.num_index == v1_yq_index, 'yr_qtr_num'].values[0]]  * (max(macro_hist.num_index) - current_yq_index + 1)
	pred_df['yq_v2_incol'] = [macro_hist.ix[macro_hist.num_index == v2_yq_index, 'yr_qtr_num'].values[0]]  * (max(macro_hist.num_index) - current_yq_index + 1)
	pred_df['yq_v3_incol'] = [macro_hist.ix[macro_hist.num_index == v3_yq_index, 'yr_qtr_num'].values[0]]  * (max(macro_hist.num_index) - current_yq_index + 1)

	pred_df['yq_pit1_incol'] = macro_hist.ix[(p1_yq_index) : (max(macro_hist.num_index) - 0), 'yr_qtr_num'].values
	pred_df['yq_pit2_incol'] = macro_hist.ix[(p2_yq_index) : (max(macro_hist.num_index) - 1), 'yr_qtr_num'].values
	pred_df['yq_pit3_incol'] = macro_hist.ix[(p3_yq_index) : (max(macro_hist.num_index) - 2), 'yr_qtr_num'].values
	 
	pred_df['gdp_v1'] = pred_df['yq_v1_incol'].map(dict(zip(macro_hist['yr_qtr_num'], macro_hist['gdp'])))
	pred_df['gdp_v2'] = pred_df['yq_v2_incol'].map(dict(zip(macro_hist['yr_qtr_num'], macro_hist['gdp'])))
	pred_df['gdp_v3'] = pred_df['yq_v3_incol'].map(dict(zip(macro_hist['yr_qtr_num'], macro_hist['gdp'])))
	pred_df['gdp_p1'] = pred_df['yq_pit1_incol'].map(dict(zip(macro_hist['yr_qtr_num'], macro_hist['gdp'])))
	pred_df['gdp_p2'] = pred_df['yq_pit2_incol'].map(dict(zip(macro_hist['yr_qtr_num'], macro_hist['gdp'])))
	pred_df['gdp_p3'] = pred_df['yq_pit3_incol'].map(dict(zip(macro_hist['yr_qtr_num'], macro_hist['gdp'])))
	pred_df['bbbspread_v1'] = pred_df['yq_v1_incol'].map(dict(zip(macro_hist['yr_qtr_num'], macro_hist['bbbspread'])))
	pred_df['bbbspread_v2'] = pred_df['yq_v2_incol'].map(dict(zip(macro_hist['yr_qtr_num'], macro_hist['bbbspread'])))
	pred_df['bbbspread_v3'] = pred_df['yq_v3_incol'].map(dict(zip(macro_hist['yr_qtr_num'], macro_hist['bbbspread'])))
	pred_df['bbbspread_p1'] = pred_df['yq_pit1_incol'].map(dict(zip(macro_hist['yr_qtr_num'], macro_hist['bbbspread'])))
	pred_df['bbbspread_p2'] = pred_df['yq_pit2_incol'].map(dict(zip(macro_hist['yr_qtr_num'], macro_hist['bbbspread'])))
	pred_df['bbbspread_p3'] = pred_df['yq_pit3_incol'].map(dict(zip(macro_hist['yr_qtr_num'], macro_hist['bbbspread']))) 
	
	pred_df['quarters_'] = np.arange((max(macro_hist.num_index) - current_yq_index + 1))

	pred_df.dropna(axis = 0, how = 'any')
	# lm.predict(pred_df.dropna(axis = 0, how = 'any'))

	## pick 25 quarters
	mean_stdev = readin_model[rating]['mean_stdev']
	anchor_point = readin_model[rating]['anchor_point']
	return readin_model[rating]['pred_model'].predict(pred_df.dropna(axis = 0, how = 'any'))[:anchor_point] * mean_stdev.ix[:(anchor_point - 1), 'd_stdev'] + mean_stdev.ix[:(anchor_point - 1), 'd_mean']

us_ci_pred(200803, 'I2to7', us_ci_model)   #pred from model
us_ci_model['I2to7']['model_fit_result'].ix[cm.yq == 200803, :] # pred from model fitting, should be the same

# compare pred vs acturl for each rating, and next 25 quarters for each starting quarter
## macro economics can only pred to 201202, which is 37th in the yq
### we can remove this part in the final calculation
def compare_pred_actual():
	tttt = pd.read_excel('H:\work\IFRS9\PCUS\CI\pc_us_cnt_cum_pd_from_sas.xlsx', sheetname = 'S1').yq
	df_out = pd.DataFrame(columns = ['pred_from_model', 'actual_data', 'pred_from_fitting', 'term_structure_mean', 'rating', 'yq'])
	for j in ['I2to7', 'S1', 'S2', 'S34', 'P123']:	
		for i in range(38):  #macro economics can only pred to 201202, which is 37th in the yq
			# the following comment out part is to predict anchor_point time long.  
			s1 = us_ci_pred(tttt[i], j, us_ci_model)   # pred anchor_point time long   
			s2 = us_ci_model[j]['model_raw_data'].ix[us_ci_model[j]['model_raw_data'].yq == tttt[i], 3:(3 + us_ci_model[j]['anchor_point'])]      
			s3 = us_ci_model[j]['model_fit_result'].ix[us_ci_model[j]['model_raw_data'].yq == tttt[i], :us_ci_model[j]['anchor_point']]     # pred from model fitting time
			s4 = us_ci_model[j]['mean_stdev'].d_mean[:us_ci_model[j]['anchor_point']]
			data1 = pd.DataFrame([s1, pd.Series(s2.values[0]), pd.Series(s3.values[0]), s4]).T
			data1.columns = ['pred_from_model', 'actual_data', 'pred_from_fitting', 'term_structure_mean'] 
			data1['rating'] = [j] * us_ci_model[j]['anchor_point']
			data1['yq'] = [tttt[i]] * us_ci_model[j]['anchor_point']
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
default_curve_data['I2to7'] = default_curve(200803, 'I2to7')
default_curve_data['S1'] = default_curve(200803, 'S1')
default_curve_data['S2'] = default_curve(200803, 'S2')
default_curve_data['S34'] = default_curve(200803, 'S34')
default_curve_data['P123'] = default_curve(200803, 'P123')

## to check why some prediction is not monotonic at the end: it is because of some point in time macro eco vars are not monotonic
###  this can be deleted after model is done
us_ci_pred(200804, 'I2to7', us_ci_model)[19:23]    # 19 to 22 is not monotonic
us_ci_model['I2to7']['pred_model'].params   # model parameters
pred_df.ix[19:22, ['yq_v3_incol', 'yq_pit1_incol', 'yq_pit3_incol', 'yq_v0_incol', 'gdp_v3', 'gdp_p1', 'bbbspread_v3', 'bbbspread_p3']]     # raw data
pred_df.ix[19:22, ['yq_v0_incol', 'gdp_v3', 'gdp_p1', 'bbbspread_v3', 'bbbspread_p3']]     # raw data
