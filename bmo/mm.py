import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import bisect

pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

mm = pd.read_excel(u"H:\python\data\hgc_us_qual.xlsx", sheetname = u'mm')
mm.drop(['Score_VolatilityMediumLongTermPerformance'], axis = 1, inplace = True)
mmfix = pd.read_excel(u"H:\\work\\usgc\\2015\\HGC_Fix_Data.xlsx", sheetname=u'mm copy')
mmfix.drop([u'EntityName', u'EntityUEN'], axis = 1, inplace = True)
mm = pd.merge(mm, mmfix, on=u'intArchiveID', how = 'left')
mm.count()  # number of non-missing records for each column
mm.isnull().sum()  # number of missing records for each column
scoreCol = [x for x in list(mm) if x.startswith('Score')]
mmscore = mm[scoreCol]
wtCol = [x.replace('Score', 'wt') for x in list(mm) if x.startswith('Score')]
maxCol = [x.replace('Score', 'max') for x in list(mm) if x.startswith('Score')]


#original wt8 was moved to 10th order because Score_VolatilityMediumLongTermPerformance is 10th order
wt = [0.1323250400, 0.1661776500, 0.1546639740, 0.0776463670, 0.1123609370, 0.0939301520, 0.1031743810, 0.0463065740, 0.0414846700, 0.0719302550] 
adj = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
wtnum = pd.Series([float(x) for x in wt])
adjnum = pd.Series([float(x) for x in adj])
	
wtCol = [x.replace('Score', 'wt') for x in list(mm) if x.startswith('Score')]

# calculate factor_score * factor_weight * weight_adjustment, df.values * Series.values * Series.values
scoreWtAdj = pd.DataFrame(mmscore.values * pd.Series(wtnum).values * pd.Series(adjnum).values, columns = mmscore.columns, index = mmscore.index)

# calculate the max_value = =IF(BN6="N/A", 0, IF(BN6>0, CM6*4,0))
scoreMax = pd.DataFrame(np.zeros(shape = mmscore.shape, dtype = float))
for i in range(len(list(mmscore))):
	scoreMax[i] = np.where(mmscore.ix[:, i] > 0, wtnum[i] * 4, 0)

scoreMax.columns = maxCol

sum1 = scoreWtAdj.sum(axis = 1)
sum2 = scoreMax.sum(axis = 1)
qca_score = sum1 / sum2
calibrated_pd = 1/(1+np.exp(-(-11.7508 + 11.7017*qca_score)))

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

mm_rank = sbc_rank = pd.concat([mm.EntityUEN, mm[u'intArchiveID'], calibrated_pd, qca_ranking, mm.FRR_rank, mm.QuantRR_rank], axis = 1)
mm_rank.columns = ['EntityUEN', 'intArchiveID', 'qualpd', 'qualrank', 'finalrank', 'quantrank']
mm_rank.ix[:, ['qualrank', 'quantrank']].pivot_table(rows= 'qualrank', cols='quantrank', aggfunc = len, fill_value = 0, margins = True)


######################################  punitive analysis  #################################
whole_rank = pd.concat([bb_rank, sbc_rank, mm_rank], axis = 0, ignore_index = True)
whole_rank.columns = ['EntityUEN', 'intArchiveID', 'qualpd', 'qualrank', 'finalrank', 'quantrank']
print pd.crosstab(whole_rank.qualrank, whole_rank.finalrank, margins = True).to_string() #show on screen without truncation

whole_rank['quant_map_pd'] = whole_rank.quantrank.replace(dict(zip(ranking, newMapPD)))
whole_rank['new30pd'] = .3 * whole_rank.qualpd + .7 * whole_rank.quant_map_pd
whole_rank['new15pd'] = .15 * whole_rank.qualpd + .85 * whole_rank.quant_map_pd

print pd.crosstab(np.digitize(whole_rank.new30pd, pdIntval), whole_rank.finalrank, margins = True)
print '-'*100
print pd.crosstab(np.digitize(whole_rank.new15pd, pdIntval), whole_rank.finalrank, margins = True)

def rating_compare(r1, r2):
	print '~'*100
	print "\t\t this is to compare " + r1.name + " v.s. " + r2.name  
	print '-'*100
	print pd.crosstab(r1, r2, margins = True).to_string()
	print '-'*100
	print r1.name + " is consistent with " + r2.name + " :   \t %s" %((r1 == r2).sum())
	print r1.name + " is different " + r2.name + ":   \t %s" %((r1 != r2).sum())
	print r1.name + " more conservative than " + r2.name + " :   \t %s" %((r1 > r2).sum())
	print r1.name + " 1 notch more than " + r2.name + " :   \t %s" %(((r1 - r2) == 1).sum())
	print r1.name + " 2 notch more than " + r2.name + " :   \t %s" %(((r1 - r2) == 2).sum())
	print r1.name + " 3+ notch more than " + r2.name + " :   \t %s" %(((r1 - r2) >2).sum())
	print r1.name + " less conservative than than " + r2.name + " :   \t %s" %((r1 < r2).sum())
	print r1.name + " 1 notch less than " + r2.name + " :   \t %s" %(((r1 - r2) == -1).sum())
	print r1.name + " 2 notch less than " + r2.name + " :   \t %s" %(((r1 - r2) == -2).sum())
	print r1.name + " 3+ notch less than " + r2.name + " :   \t %s" %(((r1 - r2) < -2).sum())
	print r1.name + " more/less " + r2.name + " :   \t %s" %(float((r1 > r2).sum()) / float((r1 < r2).sum()))
	print "within +/- 1 notch:   \t %s" %((np.abs(r1 - r2) <= 1).sum())
	print "within +/- 2 notch:   \t %s" %((np.abs(r1 - r2) <= 2).sum())
	print "+/- 3+ notch difference:   \t %s" %((np.abs(r1 - r2) > 2).sum())
	print "~"*100
	
	
	