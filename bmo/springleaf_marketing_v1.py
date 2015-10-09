import pandas as pd
import numpy as np
from sklearn import metrics

import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer
from sklearn import cross_validation
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import Lasso, SGDClassifier
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (brier_score_loss, precision_score, recall_score, f1_score)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.cross_validation import train_test_split
import timeit

 
train = pd.read_csv(r"train.csv")
test = pd.read_csv(r"test.csv")
 

# calculate AUC for each X with y
s = {}
for i in range(2, 1933):
	try:
		fpr, tpr, thresholds = metrics.roc_curve(train.ix[:, -1], train.ix[:, i])
		s[i] = metrics.auc(fpr, tpr)
	except:
		pass
		
s0 = pd.Series(s)
s0.sort()
s1 = pd.DataFrame(s0).reset_index()
s1.columns = ['iternum', '_auc']

# pick the last several highest roc vars: s1.ix[1870, 'iternum'].values
t1 = train.iloc[:, s1.ix[1870:, 'iternum'].values.tolist() + [-1]]
t1 = t1.dropna(axis = 0, how = 'any')

test_t1 = test.iloc[:, s1.ix[1870:, 'iternum'].values.tolist()]
test_t1.fillna(1, inplace = True)

x_train_1, x_valid_1, y_train_1, y_valid_1 = cross_validation.train_test_split(t1.ix[:, 0:11], t1.ix[:, 11], test_size = .4, random_state = 0)

# random forest
clf = RandomForestClassifier(n_estimators = 100)
clf.fit(x_train_1, y_train_1)
clf_pred = clf.predict(x_valid_1)
fpr, tpr, thresholds = metrics.roc_curve(clf_pred, y_valid_1)
metrics.auc(fpr, tpr)
accuracy_score(clf_pred, y_valid_1)

# test run of AdaBoostClassifier
clf = AdaBoostClassifier(n_estimators = 100, random_state = 9999)
clf.fit(x_train_1, y_train_1)
clf_pred = clf.predict(x_valid_1)
fpr, tpr, thresholds = metrics.roc_curve(clf_pred, y_valid_1)
metrics.auc(fpr, tpr)
accuracy_score(clf_pred, y_valid_1)

# GradientBoostingClassifier
clf = GradientBoostingClassifier(n_estimators = 100, random_state = 9999)
clf.fit(x_train_1, y_train_1)
clf_pred = clf.predict(x_valid_1)
fpr, tpr, thresholds = metrics.roc_curve(clf_pred, y_valid_1)
metrics.auc(fpr, tpr)
accuracy_score(clf_pred, y_valid_1)

# BaggingClassifier
clf = BaggingClassifier(n_estimators = 100, random_state = 9999)
clf.fit(x_train_1, y_train_1)
clf_pred = clf.predict(x_valid_1)
fpr, tpr, thresholds = metrics.roc_curve(clf_pred, y_valid_1)
metrics.auc(fpr, tpr)
accuracy_score(clf_pred, y_valid_1)

# ExtraTreesClassifier
clf = ExtraTreesClassifier(n_estimators = 100, random_state = 9999)
clf.fit(x_train_1, y_train_1)
clf_pred = clf.predict(x_valid_1)
fpr, tpr, thresholds = metrics.roc_curve(clf_pred, y_valid_1)
metrics.auc(fpr, tpr)
accuracy_score(clf_pred, y_valid_1)

#2.1 GridSearchCV    (very fast)
start = timeit.default_timer()
clf = SGDClassifier()
penalty_params = ['elasticnet', 'l1', 'l2']
loss_params = ['hinge', 'log']
alpha_params = [0.1, 1, 10, 50, 100]
param_grid = dict(penalty = penalty_params, loss = loss_params, alpha = alpha_params)
cv = cross_validation.KFold(t1.ix[:, 11].size, n_folds = 3, shuffle = True, random_state = 9999)
grid = GridSearchCV(clf, param_grid = param_grid, cv = cv, n_jobs = -1, scoring = 'accuracy')
grid.fit(t1.ix[:, 0:11], t1.ix[:, 11])
[x for x in grid.grid_scores_]
end = timeit.default_timer()
print end - start 

# grid search parameter C and gamma for svm rbf     very long time consuming
start = timeit.default_timer()
clf = svm.SVC(kernel='rbf')
C_range = np.logspace(-2, 5, 10)
gamma_range = np.logspace(-5, 5, 10)
param_grid = dict(C = C_range, gamma = gamma_range)
cv = cross_validation.KFold(t1.ix[:, 11].size, n_folds = 5, shuffle = True, random_state = 9999)
grid = GridSearchCV(clf, param_grid = param_grid, cv = cv, n_jobs = -1)
grid.fit(t1.ix[:, 0:11], t1.ix[:, 11])
[x for x in grid.grid_scores_]
end = timeit.default_timer()
print end - start 



### output
clf.predict(test_t1)
result = pd.DataFrame([test.ID, clf.predict(test_t1)]).T
result.columns = ['ID', 'target']
