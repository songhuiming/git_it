import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import bisect

bb = pd.read_excel(u"H:\python\data\hgc_us_qual.xlsx", sheetname = u'bb')
bb.count()  # number of non-missing records for each column
bb.isnull().sum()  # number of missing records for each column
scoreCol = [x for x in list(bb) if x.startswith('Score')]
bbscore = bb[scoreCol]
wtCol = [x.replace('Score', 'wt') for x in list(bb) if x.startswith('Score')]
maxCol = [x.replace('Score', 'max') for x in list(bb) if x.startswith('Score')]

 
wt = [0.16, 0.22, 0.16, 0.14, 0.14, 0.18]
adj = [1, 1, 1, 1, 1, 1]
wtnum = pd.Series([float(x) for x in wt])
adjnum = pd.Series([float(x) for x in adj])
	
wtCol = [x.replace('Score', 'wt') for x in list(bb) if x.startswith('Score')]

# calculate factor_score * factor_weight * weight_adjustment, df.values * Series.values * Series.values
scoreWtAdj = pd.DataFrame(bbscore.values * pd.Series(wtnum).values * pd.Series(adjnum).values, columns = bbscore.columns, index = bbscore.index)

# calculate the max_value = =IF(BN6="N/A", 0, IF(BN6>0, CM6*4,0))
scoreMax = pd.DataFrame(np.zeros(shape = bbscore.shape, dtype = float))
for i in range(len(list(bbscore))):
	scoreMax[i] = np.where(bbscore.ix[:, i] > 0, wtnum[i] * 4, 0)

scoreMax.columns = maxCol

sum1 = scoreWtAdj.sum(axis = 1)
sum2 = scoreMax.sum(axis = 1)
qca_score = sum1 / sum2
calibrated_pd = 1/(1+np.exp(-(-13.6264+15.5874*qca_score)))

## after pd is got, it will be mapped to master scale rating, ranking, and 
# bin to interval and map to a given label
pdIntval = [0.0003, 0.0007, 0.0011, 0.0019, 0.0032, 0.0054, 0.0091, 0.0154, 0.0274, 0.0516, 0.097, 0.1823, 1]
msRating = ['I-2', 'I-3', 'I-4', 'I-5', 'I-6', 'I-7', 'S-1', 'S-2', 'S-3', 'S-4', 'P-1', 'P-2', 'P-3', 'D-1']
pcRR = [10, 15, 20, 25, 30, 35, 40, 45, 46, 47, 50, 51, 52]
ranking = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
newMapPD = [0.0003, 0.0005, 0.0009, 0.0015, 0.0026, 0.0043, 0.0073, 0.0123, 0.0214, 0.0395, 0.0743, 0.1397, 0.2412, 1]
oldMapPD = [0.0004, 0.0007, 0.0013, 0.0025, 0.004, 0.0067, 0.0107, 0.018, 0.0293, 0.0467, 0.0933, 0.18, 0.3333, 1]

#build mapping dict
rank2msr = dict(zip(ranking, msRating))
msr2rank = dict(zip(msRating, ranking))
rank2pcrr = dict(zip(ranking, pcRR))
pcrr2rank = dict(zip(pcRR, ranking))
rank2pd = dict(zip(ranking, newMapPD))

# 0.00312 -->  bin interval is (0.0019, 0.0032)  -->  ranking = 6
def pd2rank(x):
	pdIntval = [0.0003, 0.0007, 0.0011, 0.0019, 0.0032, 0.0054, 0.0091, 0.0154, 0.0274, 0.0516, 0.097, 0.1823, 1]
	ranking = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
	return ranking[bisect.bisect_left(pdIntval, x)] #map pd to order, then pick ranking[order]

qca_ranking = pd.Series(map(pd2rank, calibrated_pd))  #same as   np.digitize(calibrated_pd, pdIntval)
qca_msRank = qca_ranking.apply(lambda x: rank2msr.get(x, x)) #same as    pd.Series(map(lambda x: rank2msr.get(x, ), qca_ranking))

bb_rank = bb_rank = pd.concat([bb.EntityUEN, bb[u'intArchiveID'], calibrated_pd, qca_ranking, bb.FRR_rank, bb.QuantRR_rank], axis = 1)
bb_rank.columns = ['EntityUEN', 'intArchiveID', 'qualpd', 'qualrank', 'finalrank', 'quantrank']
