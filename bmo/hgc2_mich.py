#!/usr/bin/python

## compate Miche result with Final Rating

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# read in data
hgc2 = pd.read_excel(u"H:\\work\\usgc\\2015\\HGC_f2014_GCRating_Michelle.xlsx", header = 0, sheet_name="sheet1")

pdIntval = [0.0003, 0.0007, 0.0011, 0.0019, 0.0032, 0.0054, 0.0091, 0.0154, 0.0274, 0.0516, 0.097, 0.1823, 1]
msRating = ['I-2', 'I-3', 'I-4', 'I-5', 'I-6', 'I-7', 'S-1', 'S-2', 'S-3', 'S-4', 'P-1', 'P-2', 'P-3', 'D-1']
pcRR = [10, 15, 20, 25, 30, 35, 40, 45, 46, 47, 50, 51, 52]
ranking = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
newMapPD = [0.0003, 0.0005, 0.0009, 0.0015, 0.0026, 0.0043, 0.0073, 0.0123, 0.0214, 0.0395, 0.0743, 0.1397, 0.2412]
oldMapPD = [0.0004, 0.0007, 0.0013, 0.0025, 0.004, 0.0067, 0.0107, 0.018, 0.0293, 0.0467, 0.0933, 0.18, 0.3333]

hgc2['p_quant_rank'] = hgc2.ProposGC_QuanRR.replace(dict(zip(msRating, ranking)))
pd.crosstab(hgc2.p_quant_rank, hgc2.ProposGC_QuanRR.values)

#need to read in whole_rank data before analysis
hgcmerge = pd.merge(hgc2, whole_rank, on = u'intArchiveID', how = 'inner')
rating_compare(hgcmerge.qualrank, hgcmerge.p_quant_rank)

#compare the weighted qual+quant v.s. proposed_quant_rank
def wt_compare(wt):
	hgcmerge['new' + str(wt) + 'rank'] = np.round(hgcmerge.qualrank * (1.0 - float(wt)/100) + hgcmerge.quantrank * float(wt)/100, 0)
	rating_compare(hgcmerge['new' + str(wt) + 'rank'], hgcmerge.p_quant_rank)

wt_compare(15)
wt_compare(30)

