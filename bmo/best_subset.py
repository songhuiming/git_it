import pandas as pd
import numpy as np
from itertools import chain, combinations
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm
from statsmodels.formula.api import logit, glm

hgcdev = pd.read_excel(u"H:\\work\\usgc\\2015\\quant\\hgcdev0311g_shm.xlsx", sheetname = "dev4py")
quant_var = [ u'Debt_Service_Coverage', u'cash_security_to_curLiab',  u'Debt_to_Tangible_Net_Worth',  u'Yrs_In_Business',  u'Total_Sales', u'Debt_to_EBITDA']
quant_var2 = [ u'EBIT',  u'EBITDA',  u'TNW',  u'Cash_n_Securities']
hgc = hgcdev.ix[:, quant_var+['TNW', 'DF']].dropna(axis = 0)
hgcvar = list(hgc)
f = hgcvar[-1] + ' ~ ' + ' + '.join(hgcvar[:-1])
logfit = logit(formula = str(f), data = hgc).fit()
print logfit.summary()    
logfit.params  #get regression coefficient estimation
logfit.fittedvalues   # log odds , log(p/(1-p))
logfit.predict()  # predicted probability
logfit.pred_table()  # confusion matrix , np.array([['actual=1, pred=1', 'actual=0, pred=1'], ['actual=0, pred = 1', 'actual=0, pred=0']])
## ROC/AUC
def auc(x, y):
	unq_x = np.unique(x)
	n1 = sum(y)
	n = len(y)
	Sens = np.zeros_like(unq_x)
	Spec = np.zeros_like(unq_x)
	for j, u in enumerate(unq_x):
		Sens[j] = np.sum((x >= u) * y) / float(n1)
		Spec[j] = np.sum((x <= u) *(1 - y)) / float(n - n1)
	auc = 0.0
	for i in range(len(Spec) - 1):
		auc += (Spec[i + 1] - Spec[i]) * (Sens[i + 1] + Sens[i]) / 2.0
	return auc

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

estimator = RandomForestRegressor(random_state=0, n_estimators=100)
best_subset_cv(estimator, X.values[:, range(6)], X.values[:, 6])

##############  test logistic regression  ############
from sklearn import linear_model
logistic = linear_model.LogisticRegression(C = 100000, weight = None)
#ldata = hgcdev.ix[:, ['DF'] + quant_var+['TNW']].dropna(axis = 0)
#logresult = logistic.fit(ldata.values[:, np.arange(7)+1], ldata.values[:,0])
ldata = hgcdev.ix[:, ['DF', u'Debt_Service_Coverage', u'cash_security_to_curLiab']].dropna(axis = 0)
logresult = logistic.fit(ldata.values[:, [1, 2]], ldata.values[:,0])

## in R
df = read.csv("H:\\work\\usgc\\2015\\quant\\test_4_r.csv")
glm(DF ~ Debt_Service_Coverage + cash_security_to_curLiab, data = df, family = "binomial")

##############  test linear regression, same result as SAS output  ############
lr = linear_model.LinearRegression()
lrresult = lr.fit(ldata.values[:, [1, 2]], ldata.values[:, 3])