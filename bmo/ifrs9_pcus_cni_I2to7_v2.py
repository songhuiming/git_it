import pandas as pd
import numpy as np
import itertools
from itertools import chain, combinations
import statsmodels.formula.api as smf
import scipy.stats as stats
from scipy.stats import norm
import statsmodels.api as sm
import matplotlib.pyplot as plt
%matplotlib inline 
plt.rcParams['figure.figsize'] = (16, 12)
# combine I2 to I7 together, and normalize data, then run regression
## v2: udpate 20151007: update to get the anchor point from mean value, and adjust the upper triangle to vector part
  
#########################################   Part I:  Data Preparation   #############################################

cm = pd.read_excel('H:\work\IFRS9\PCUS\CI\pc_us_cnt_cum_pd_from_sas.xlsx', sheetname = 'I2to7')   # cum pd from sas, need to normalize it
cumcol = [x for x in cm.columns if x.startswith('cumpd')]
print cm.ix[:, cumcol].shape
cm_cum = cm.ix[:, cumcol]

mean_stdev = pd.DataFrame(np.array([np.nanmean(cm_cum, axis = 0), np.nanstd(cm_cum, axis = 0)]).T, columns = ['d_mean', 'd_stdev'])
anchor_point = sum(mean_stdev.d_mean.diff().fillna(method = 'backfill') > 0)

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
yr_qtr['yr_qtr_str'] = [str(x) for x in yr_qtr.yr_qtr_num]
# macro_hist['yr_qtr_str'] = [str(x) for x in macro_hist.yr_qtr_num]

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
#########################################################################################################################################

# df.to_excel('H:\work\IFRS9\CM\cm_pd_qtrbyqtr_w_macro.xlsx', sheet_name = 'I-7', index = False)


#########################################   Part II: Modeling Part   #############################################
# f = 'cum_pd_num~gdp_v1+gdp_v2+gdp_v3+gdp_p1+gdp_p2+gdp_p3+unemploy_rate_v1+unemploy_rate_v2+unemploy_rate_v3+unemploy_rate_p1+unemploy_rate_p2+unemploy_rate_p3'
# f = 'cum_pd_num~gdp_v1+gdp_v2+gdp_v3+gdp_p1+gdp_p2+gdp_p3+bbbspread_v1+bbbspread_v2+bbbspread_v3+bbbspread_p1+bbbspread_p2+bbbspread_p3'

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

#f = 'cum_pd_num ~ gdp_v3 + gdp_p2 + bbbspread_v3 + quarters_'	 
f = res_df.ix[0, 'f']
lm = smf.ols(formula = str(f), data = df).fit()
print lm.summary()

# for plot
# np1 = np.array([lm.predict(), df.cum_pd_num]).T
# plt.plot(np1[np1[:, 1].argsort()])

df1 = pd.DataFrame(np.array([lm.predict(), df.cum_pd_num]).T, columns = ['predicted', 'y'])
plt.plot(df1.sort('predicted'))


# from column to upper triangle matirx: 775 to cm_cum shape
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
#################################################################################################################################

 
 
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

 

#############################################  test code for "200803" PD predict    #########################################################  

# to predict for a give time,   be careful with   "200803"   to replace
# 大概： 对我们要predict的那个year qtr， 比如 ‘200803’， 找出它的 v1 v2 v3 时间 , 分别为(‘200802’ ‘200801’ ‘200704’)

yr_qtr = pd.read_excel(r'H:\work\IFRS9\PCUS\CI\200803_current_pit_macroeconomics.xlsx', sheetname = 'yr_qtr')
macro_hist = pd.read_excel(r'H:\work\IFRS9\PCUS\CI\200803_current_pit_macroeconomics.xlsx', sheetname = 'macro_hist') 
yr_qtr['yr_qtr_str'] = [str(x) for x in yr_qtr.yr_qtr_num]
macro_hist['yr_qtr_str'] = [str(x) for x in macro_hist.yr_qtr_num]

current_yq_index = yr_qtr.ix[yr_qtr.yr_qtr_str == "200803", 'num_index'].values[0]    # '200803' will be 35
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
lm.predict(pred_df.dropna(axis = 0, how = 'any'))

## pick 25 quarters
print lm.predict(pred_df.dropna(axis = 0, how = 'any'))[:25] * mean_stdev.ix[:24, 'd_stdev'] + mean_stdev.ix[:24, 'd_mean']

plt.plot(lm.predict(pred_df.dropna(axis = 0, how = 'any'))[:25] * mean_stdev.ix[:24, 'd_stdev'] + mean_stdev.ix[:24, 'd_mean'], mean_stdev.ix[:24, 'd_mean'])













############################################################  Stop to Copy Here  ################################################################





# plot consecutive 12 quarters actual v.s. predicted value from each start point
plt.plot(range(1, 13), ypred_to_tri.ix[17, 1:13], label = r'pred')
plt.plot(range(1, 13), cm_cum.ix[17, 1:13], label = r'true')
 

# plot pred mean v.s. actual mean for the 25 quarters
plt.plot(range(1, 26), ypred_to_tri.mean(axis = 0)[:25])
plt.plot(range(1, 26), cm_cum.mean(axis = 0)[:25])
 
xvar = 'gdp_v1+gdp_v2+gdp_v3+gdp_p1+gdp_p2+gdp_p3+bbbspread_v1+bbbspread_v2+bbbspread_v3+bbbspread_p1+bbbspread_p2+bbbspread_p3+quarters_'.split('+')
X = df.ix[:, xvar]
y = [float(x) for x in df.ix[:, 'cum_pd']]

# list all combination of the X and run reg on each model, totally there are 
def permunation_x_reg(X, y):
	res = {}
	n_features = X.shape[1]
	subsets = chain.from_iterable(combinations(xrange(n_features), k+1) for k in xrange(n_features))
	best_score = -np.inf
	best_subset = None
	try:
		for subset in subsets:
			#xwqtr = pd.concat([X.iloc[:, subset], df.quarters_], axis = 1)
			#lin_reg = sm.OLS(y, xwqtr).fit()
			lin_reg = sm.OLS(y, X.iloc[:, subset]).fit()
			score = lin_reg.rsquared_adj
			res[subset] = score
	except:
		pass
	return res

res = permunation_x_reg(X, y) 
import operator
sorted_res = sorted(res.items(), key = operator.itemgetter(1), reverse = True)

# get the max rsq regression with 6 variables
ps1 = pd.DataFrame(sorted_res, columns = ['xorder', 'rsq'])
ps1['lens'] = ps1.xorder.map(lambda x: len(x))
ps1.head()
ps1[ps1.lens == 6]



# find best subset regression based on all combinations
def best_subset(X, y):
	n_features = X.shape[1]
	subsets = chain.from_iterable(combinations(xrange(n_features), k+1) for k in xrange(n_features))
	best_score = -np.inf
	best_subset = None
	for subset in subsets:
		lin_reg = sm.OLS(y, X.iloc[:, subset]).fit()
		score = lin_reg.rsquared_adj
		if score > best_score:
			best_score, best_subset = score, subset
	return best_subset, best_score

best_subset(X, y) 

