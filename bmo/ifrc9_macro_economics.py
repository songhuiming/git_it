import pandas as pd
import numpy as np
import itertools
from itertools import chain, combinations
import statsmodels.formula.api as smf
import scipy.stats as stats
from scipy.stats import norm
import statsmodels.api as sm
 

  

cm = pd.read_excel('H:\work\IFRS9\CM\cm_pd_qtrbyqtr.xlsx', sheetname = 'I-7')
cumcol = [x for x in cm.columns if x.startswith('cumpd')]
cm.ix[:, cumcol].shape
cm_cum = cm.ix[:, cumcol]

#按行拉直
cumpd_incol = [cm_cum.iloc[i, j] for i in range(43) for j in range(43 - i)]     # len=946

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

yq_pit0_incol = [str(x) for ll in [yq_p0[i:] for i in range(43)] for x in ll] 		# len=946
yq_pit1_incol = [str(x) for ll in [yq_p1[i:] for i in range(43)] for x in ll]
yq_pit2_incol = [str(x) for ll in [yq_p2[i:] for i in range(43)] for x in ll]
yq_pit3_incol = [str(x) for ll in [yq_p3[i:] for i in range(43)] for x in ll]

# set up vintage time
yq_v0 = gdp.ix[4:46, 'yq']     	#from 2003Q1 to 2013Q3
yq_v1 = gdp.ix[3:45, 'yq']		#from 2002Q4 to 2013Q2
yq_v2 = gdp.ix[2:44, 'yq']
yq_v3 = gdp.ix[1:43, 'yq']

yq_v0_incol = [yq_v0.iloc[i] for i in range(43) for j in range(43 - i)]
yq_v1_incol = [yq_v1.iloc[i] for i in range(43) for j in range(43 - i)]
yq_v2_incol = [yq_v2.iloc[i] for i in range(43) for j in range(43 - i)]
yq_v3_incol = [yq_v3.iloc[i] for i in range(43) for j in range(43 - i)]


df = pd.DataFrame([yq_v0_incol, yq_v1_incol, yq_v2_incol, yq_v3_incol, yq_pit0_incol, yq_pit1_incol, yq_pit2_incol, yq_pit3_incol, cumpd_incol]).T
df.columns = ['yq_v0_incol', 'yq_v1_incol', 'yq_v2_incol', 'yq_v3_incol', 'yq_pit0_incol', 'yq_pit1_incol', 'yq_pit2_incol', 'yq_pit3_incol', 'cum_pd']

df['inverse_cpd'] = (df.cum_pd + 10E-6).map(lambda x: norm.ppf(x))

# join data to get macro economics info
#1 gdp
df['gdp_v1'] = df.yq_v1_incol.map(dict(zip(gdp['yq'], gdp['gdp'])))
df['gdp_v2'] = df.yq_v2_incol.map(dict(zip(gdp['yq'], gdp['gdp'])))
df['gdp_v3'] = df.yq_v3_incol.map(dict(zip(gdp['yq'], gdp['gdp'])))

df['gdp_p1'] = df.yq_pit1_incol.map(dict(zip(gdp['yq'], gdp['gdp'])))
df['gdp_p2'] = df.yq_pit2_incol.map(dict(zip(gdp['yq'], gdp['gdp'])))
df['gdp_p3'] = df.yq_pit3_incol.map(dict(zip(gdp['yq'], gdp['gdp'])))
#2 bbb bond yield rate
df['bbb_v1'] = df.yq_v1_incol.map(dict(zip(gdp['yq'], gdp['bbb'])))
df['bbb_v2'] = df.yq_v2_incol.map(dict(zip(gdp['yq'], gdp['bbb'])))
df['bbb_v3'] = df.yq_v3_incol.map(dict(zip(gdp['yq'], gdp['bbb'])))

df['bbb_p1'] = df.yq_pit1_incol.map(dict(zip(gdp['yq'], gdp['bbb'])))
df['bbb_p2'] = df.yq_pit2_incol.map(dict(zip(gdp['yq'], gdp['bbb'])))
df['bbb_p3'] = df.yq_pit3_incol.map(dict(zip(gdp['yq'], gdp['bbb'])))

df.to_excel('H:\work\IFRS9\CM\cm_pd_qtrbyqtr_w_macro.xlsx', sheet_name = 'I-7', index = False)

# f = 'inverse_cpd~gdp_v1+gdp_v2+gdp_v3+gdp_p1+gdp_p2+gdp_p3+unemploy_rate_v1+unemploy_rate_v2+unemploy_rate_v3+unemploy_rate_p1+unemploy_rate_p2+unemploy_rate_p3'
f = 'inverse_cpd~gdp_v1+gdp_v2+gdp_v3+gdp_p1+gdp_p2+gdp_p3+bbb_v1+bbb_v2+bbb_v3+bbb_p1+bbb_p2+bbb_p3'
lm = smf.ols(formula = str(f), data = df).fit()
print lm.summary()

f = 'inverse_cpd~ gdp_v2 + gdp_p2 + bbb_v2 + bbb_p2'
lm = smf.ols(formula = str(f), data = df).fit()
print lm.summary()

f = 'inverse_cpd~ gdp_v2 + bbb_v2 + bbb_p2'
lm = smf.ols(formula = str(f), data = df).fit()
print lm.summary()



def best_subset(X, y):
	n_features = X.shape[1]
	subsets = chain.from_iterable(combinations(xrange(n_features), k+1) for k in xrange(n_features))
	best_score = -np.inf
	best_subset = None
	for subset in subsets:
		logistic_reg = sm.Logit(y, X.iloc[:, subset]).fit()
		pred = logistic_reg.predict()
		score = auc(pred, y)
		if score > best_score:
			best_score, best_subset = score, subset
	return best_subset, best_score

 