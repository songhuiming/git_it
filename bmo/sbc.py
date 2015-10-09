import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import bisect

sbc = pd.read_excel(u"H:\python\data\hgc_us_qual.xlsx", sheetname = u'sbc')
sbc.count()  # number of non-missing records for each column
sbc.isnull().sum()  # number of missing records for each column
scoreCol = [x for x in list(sbc) if x.startswith('Score')]
sbcscore = sbc[scoreCol]
wtCol = [x.replace('Score', 'wt') for x in list(sbc) if x.startswith('Score')]
maxCol = [x.replace('Score', 'max') for x in list(sbc) if x.startswith('Score')]

wt0 = u'0.166666666   0.166666666     0       0       0.166666667     0.166666667     0       0.166666667     0.166666667'     # copied from excel
wt = [float(x) for x in wt0.split()]
# wt = ['0.166666666', '0.166666666', '0', '0', '0.166666667', '0.166666667', '0', '0.166666667', '0.166666667']
adj = ['1', '1', '1', '1.333333', '1', '1', '1', '1.333333', '1']
wtnum = pd.Series([float(x) for x in wt])
adjnum = pd.Series([float(x) for x in adj])
	
wtCol = [x.replace('Score', 'wt') for x in list(sbc) if x.startswith('Score')]

# calculate factor_score * factor_weight * weight_adjustment, df.values * Series.values * Series.values
scoreWtAdj = pd.DataFrame(sbcscore.values * pd.Series(wtnum).values * pd.Series(adjnum).values, columns = sbcscore.columns, index = sbcscore.index)

# calculate the max_value = =IF(BN6="N/A", 0, IF(BN6>0, CM6*4,0))
scoreMax = pd.DataFrame(np.zeros(shape = sbcscore.shape, dtype = float))
for i in range(len(list(sbcscore))):
	scoreMax[i] = np.where(sbcscore.ix[:, i] > 0, wtnum[i] * 4, 0)

scoreMax.columns = maxCol

sum1 = scoreWtAdj.sum(axis = 1)
sum2 = scoreMax.sum(axis = 1)
qca_score = sum1 / sum2
calibrated_pd = 1/(1+np.exp(-(-11.54+13.399*qca_score)))

## after pd is got, it will be mapped to master scale rating, ranking, and 
# bin to interval and map to a given label
pdIntval = [0.0003, 0.0007, 0.0011, 0.0019, 0.0032, 0.0054, 0.0091, 0.0154, 0.0274, 0.0516, 0.097, 0.1823, 1]
msRating = ['I-2', 'I-3', 'I-4', 'I-5', 'I-6', 'I-7', 'S-1', 'S-2', 'S-3', 'S-4', 'P-1', 'P-2', 'P-3', 'D-1']
pcRR = [10, 15, 20, 25, 30, 35, 40, 45, 46, 47, 50, 51, 52]
ranking = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
newMapPD = [0.0003, 0.0005, 0.0009, 0.0015, 0.0026, 0.0043, 0.0073, 0.0123, 0.0214, 0.0395, 0.0743, 0.1397, 0.2412]
oldMapPD = [0.0004, 0.0007, 0.0013, 0.0025, 0.004, 0.0067, 0.0107, 0.018, 0.0293, 0.0467, 0.0933, 0.18, 0.3333]

#build mapping dict
rank2msr = dict(zip(ranking, msRating))
msr2rank = dict(zip(msRating, ranking))
rank2pcrr = dict(zip(ranking, pcRR))
pcrr2rank = dict(zip(pcRR, ranking))

# 0.00312 -->  bin interval is (0.0019, 0.0032)  -->  ranking = 6
def pd2rank(x):
	pdIntval = [0.0003, 0.0007, 0.0011, 0.0019, 0.0032, 0.0054, 0.0091, 0.0154, 0.0274, 0.0516, 0.097, 0.1823, 1]
	ranking = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
	return ranking[bisect.bisect_left(pdIntval, x)] #map pd to order, then pick ranking[order]

qca_ranking = pd.Series(map(pd2rank, calibrated_pd))  #same as   np.digitize(calibrated_pd, pdIntval)
qca_msRank = qca_ranking.apply(lambda x: rank2msr.get(x, x)) #same as    pd.Series(map(lambda x: rank2msr.get(x, ), qca_ranking))

sbc_rank = sbc_rank = pd.concat([sbc.EntityUEN, sbc[u'intArchiveID'], calibrated_pd, qca_ranking, sbc.FRR_rank, sbc.QuantRR_rank], axis = 1)
sbc_rank.columns = ['EntityUEN', 'intArchiveID', 'qualpd', 'qualrank', 'finalrank', 'quantrank']
