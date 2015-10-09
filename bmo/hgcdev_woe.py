#hgc v4 replicate

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import bisect

hgcdev = pd.read_excel(u"H:\\work\\usgc\\2015\\quant\\hgcdev0311g_shm.xlsx", sheetname = "dev4py")

quant_var = [ u'Debt_Service_Coverage', u'cash_security_to_curLiab',  u'Debt_to_Tangible_Net_Worth',  u'Yrs_In_Business',  u'Total_Sales', u'Debt_to_EBITDA']
quant_var2 = [ u'EBIT',  u'EBITDA',  u'TNW',  u'Cash_n_Securities']

#1 DSC
dsc = [0, 1, 1.25, 2.06452, 4.45]
dsc_woe = [-1.33433, -0.99743, -0.86373, 0.34456, 0.77557, 1.67578]

#2 cash_security_to_curLiab
cash_security_2_curLiab = [0.01821, 0.12446, 0.37931, 1.15772]
cash_security_2_curLiab_woe = [-1.03004, -0.31255, 0.30191, 0.76281, 1.70542]

#3 Debt_to_Tangible_Net_Worth
debt_2_tnw = [0, 0.61247, 1.28134, 2.48821, 4.55724] 
debt_2_tnw_woe = [-1.06007, 1.32242, 1.12546, 0.52609, 0.21725, -0.75127]

#4 years in business
yrs_in_b = [3, 10, 16.41667, 24.89589, 39.23836]
yrs_in_b_woe = [-1.11391, -0.46495, -0.24602, -0.03031, 0.61446, 1.04355]

#5 Total_Sales
tot_sales = [5000000, 20000000]
tot_sales_woe = [-0.56285, 0.46009, 1.22319]

#6 Debt_to_EBITDA
dt_2_ebitda = [0, 2, 4.54977, 6.34321, 9.90462, 14.93103]
dt_2_ebitda_woe = [-1.27817, 1.40998, 1.00044, 0.43944, 0.20636, -0.604078, -0.82749]


def woe(x, bin_x, woe_value, right=1):
	return [woe_value[i] for i in np.digitize(np.array(x), np.array(bin_x), right = right)]

#1 logic: dsc=na, _dsc=0; dsc=0, _dsc=-0.99743; the rest are right included; ebit<0, go to bad;
hgcdev['_dsc'] = woe(hgcdev.ix[:, u'Debt_Service_Coverage'], dsc, dsc_woe)
hgcdev[u'_dsc'][hgcdev[u'Debt_Service_Coverage'].isnull()] = 0
hgcdev[u'_dsc'][hgcdev[u'Debt_Service_Coverage'] == 0] = -0.99743
hgcdev[u'_dsc'][(hgcdev[u'EBIT'] < 0) & (hgcdev[u'Debt_Service_Coverage'].isnull())] = -1.33433
hgcdev[u'_dsc'].value_counts(dropna = 0).sort_index()
# dsc2 = [-inf, 0, 1, 1.25, 2.06452, 4.45, inf]
# pd.value_counts(pd.cut(hgcdev[u'_dsc'], dsc2, right = 1), sort=1, dropna = 0)

#2: (Cash_n_Securities<=0 or na) & cash_security_to_curLiab: _cash_security_2_curLiab = -1.03004
hgcdev['_cash_security_2_curLiab'] = woe(hgcdev.ix[:, u'cash_security_to_curLiab'], cash_security_2_curLiab, cash_security_2_curLiab_woe)
hgcdev['_cash_security_2_curLiab'][hgcdev[u'cash_security_to_curLiab'].isnull()] = 0
hgcdev['_cash_security_2_curLiab'][((hgcdev[u'Cash_n_Securities'] <= 0) | (hgcdev[u'Cash_n_Securities'].isnull())) & (hgcdev[u'cash_security_to_curLiab'].isnull())] = -1.03004
hgcdev['_cash_security_2_curLiab'].value_counts(dropna = 0).sort_index()

#3: d_2_tnw=0: _d_2_tnw=1.32242; d_2_tnw=na & tnw<=0: _d_2_tnw=-1.06007  
hgcdev[u'_debt_2_tnw'] = woe(hgcdev.ix[:, u'Debt_to_Tangible_Net_Worth'], debt_2_tnw, debt_2_tnw_woe)
hgcdev[u'_debt_2_tnw'][hgcdev[u'Debt_to_Tangible_Net_Worth'].isnull()] = 0
hgcdev[u'_debt_2_tnw'][hgcdev[u'Debt_to_Tangible_Net_Worth'] == 0] = 1.32242
hgcdev[u'_debt_2_tnw'][(hgcdev[u'Debt_to_Tangible_Net_Worth'].isnull()) & (hgcdev[u'TNW'] <=0)] = -1.06007 
hgcdev[u'_debt_2_tnw'].value_counts().sort_index()

#4: yr_in_b = na = -1.11391, as sas code: if Yrs_In_B<0 then _Yrs_In_B=-1.11391;
hgcdev[u'_yrs_in_b'] = woe(hgcdev.ix[:, u'Yrs_In_Business'], yrs_in_b, yrs_in_b_woe)
hgcdev[u'_yrs_in_b'][hgcdev[u'Yrs_In_Business'].isnull()] = -1.11391
hgcdev[u'_yrs_in_b'][hgcdev[u'Yrs_In_Business'] == 3] = -1.11391
hgcdev[u'_yrs_in_b'][hgcdev[u'Yrs_In_Business'] < 0] = -1.11391
hgcdev[u'_yrs_in_b'].value_counts().sort_index()

#5: left included in sas: low-<5000000   5000000-<20000000   20000000-High
hgcdev[u'_tot_sales'] = woe(hgcdev[u'Total_Sales'], tot_sales, tot_sales_woe, right = 0)
hgcdev[u'_tot_sales'][hgcdev[u'Total_Sales'].isnull()] = -0.56285;
hgcdev[u'_tot_sales'].value_counts().sort_index()

#6: 
hgcdev[u'_dt_2_ebitda'] = woe(hgcdev[u'Debt_to_EBITDA'], dt_2_ebitda, dt_2_ebitda_woe)
hgcdev[u'_dt_2_ebitda'][hgcdev[u'Debt_to_EBITDA'] == 0] = 1.40998
hgcdev[u'_dt_2_ebitda'][hgcdev[u'Debt_to_EBITDA'].isnull()] = 0
hgcdev[u'_dt_2_ebitda'][(hgcdev[u'Debt_to_EBITDA'].isnull()) & (hgcdev[u'EBITDA'] <= 0)] = -1.27817
hgcdev[u'_dt_2_ebitda'].value_counts().sort_index()

[hgcdev[x].value_counts().sort_index() for x in list(hgcdev) if x.startswith('_')]

#[('_dsc', -0.5566), ('_cash_security_2_curLiab', -0.7271), (u'_yrs_in_b', -0.6119), (u'_tot_sales', -0.7048), (u'_dt_2_ebitda', -0.3473), (u'_debt_2_tnw', -0.4334)]

coefVec = np.array([-0.5566, -0.7271, -0.6119, -0.7048, -0.3473, -0.4334])
hgcdev['odd1'] = -3.4147 + np.sum(np.array([hgcdev[x] for x in list(hgcdev) if x.startswith('_')]).T * coefVec, axis = 1)

#output _xxx_vars (var2woe values) into excel
#pd.DataFrame(pd.concat([pd.Series(hgcdev[x]) for x in list(hgcdev) if x.startswith('_')], axis = 1)).to_excel(u"H:\\work\\usgc\\2015\\quant\\hgcdev_calc.xlsx", sheet_name="_vars")

 








