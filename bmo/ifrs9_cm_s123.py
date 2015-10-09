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
  

cm = pd.read_excel('H:\work\IFRS9\CM\qtr_summary_cm_cm_20150921.xlsx', sheetname = 'S123')   # normalized cum pd
cumcol = [x for x in cm.columns if x.startswith('cumpd')]
print cm.ix[:, cumcol].shape
cm_cum = cm.ix[:, cumcol]

#左上三角矩阵按行拉直 			# len=775
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

cumpd_incol = tri_to_line(cm_cum)    

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

# set up vintage time
yq_v0 = gdp.ix[4:46, 'yq']     	#from 2003Q1 to 2013Q3
yq_v1 = gdp.ix[3:45, 'yq']		#from 2002Q4 to 2013Q2
yq_v2 = gdp.ix[2:44, 'yq']
yq_v3 = gdp.ix[1:43, 'yq']

def vintage_data(indata):
	outdata = []
	for i in range(43):
		if i <= 18:
			for j in range(25):
				outdata.append(indata.iloc[i])
		else:
			for j in range(43 - i):
				outdata.append(indata.iloc[i])
	return outdata

yq_v0_incol = vintage_data(yq_v0)   
yq_v1_incol = vintage_data(yq_v1)
yq_v2_incol = vintage_data(yq_v2)
yq_v3_incol = vintage_data(yq_v3)

df = pd.DataFrame([yq_v0_incol, yq_v1_incol, yq_v2_incol, yq_v3_incol, yq_pit0_incol, yq_pit1_incol, yq_pit2_incol, yq_pit3_incol, cumpd_incol]).T
df.columns = ['yq_v0_incol', 'yq_v1_incol', 'yq_v2_incol', 'yq_v3_incol', 'yq_pit0_incol', 'yq_pit1_incol', 'yq_pit2_incol', 'yq_pit3_incol', 'cum_pd']

df['inverse_cpd'] = (df.cum_pd + 10E-6).map(lambda x: norm.ppf(x))
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
	
# df.to_excel('H:\work\IFRS9\CM\cm_pd_qtrbyqtr_w_macro.xlsx', sheet_name = 'I-7', index = False)

# f = 'cum_pd_num~gdp_v1+gdp_v2+gdp_v3+gdp_p1+gdp_p2+gdp_p3+unemploy_rate_v1+unemploy_rate_v2+unemploy_rate_v3+unemploy_rate_p1+unemploy_rate_p2+unemploy_rate_p3'

f = 'cum_pd_num~gdp_v1+gdp_v2+gdp_v3+gdp_p1+gdp_p2+gdp_p3+bbbspread_v1+bbbspread_v2+bbbspread_v3+bbbspread_p1+bbbspread_p2+bbbspread_p3'
lm = smf.ols(formula = str(f), data = df).fit()
print lm.summary()

f = 'cum_pd_num~ gdp_v1 + gdp_p3 + bbbspread_v3 + bbbspread_p3 + quarters_'
lm = smf.ols(formula = str(f), data = df).fit()
print lm.summary()

f = 'cum_pd_num~ gdp_v1 + gdp_v2 + gdp_p1 + bbbspread_v1 + bbbspread_v3 + quarters_'     # rsq = 0.660
lm = smf.ols(formula = str(f), data = df).fit()
print lm.summary()

f = 'cum_pd_num~ gdp_v1 + gdp_v2 + gdp_p1 + bbbspread_v1 +  quarters_'		# 0.640
lm = smf.ols(formula = str(f), data = df).fit()
print lm.summary()

# for plot
# np1 = np.array([lm.predict(), df.cum_pd_num]).T
# plt.plot(np1[np1[:, 1].argsort()])

df1 = pd.DataFrame(np.array([lm.predict(), df.cum_pd_num]).T, columns = ['predicted', 'y'])
plt.plot(df1.sort('predicted'))


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

mean_stdev = pd.read_excel('H:\work\IFRS9\CM\qtr_summary_cm_cm_20150921.xlsx', sheetname = 'S123_mean_stdev')
for i in range(25):
	cm_cum.ix[:, i] = cm_cum.ix[:, i] * mean_stdev.ix[i, 'd_stdev'] + mean_stdev.ix[i, 'd_mean']
	ypred_to_tri.ix[:, i] = ypred_to_tri.ix[:, i] * mean_stdev.ix[i, 'd_stdev'] + mean_stdev.ix[i, 'd_mean']



	
	
fig = plt.figure(figsize = (16, 12))
ax = fig.add_subplot(111)
 
ax.plot(range(1, 26), ypred_to_tri.mean(axis = 0)[:25], color = 'r', linestyle = '-', linewidth = 2, markeredgecolor='none', marker = '', label = r'pred avg')
ax.plot(range(1, 26), cm_cum.mean(axis = 0)[:25], color = 'b', linestyle = '-', linewidth = 2, markeredgecolor='none', marker = '', label = r'true avg')
plt.legend(loc = 3, ncol = 1) 
	
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

