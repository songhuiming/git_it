
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import bisect

hgcdev_all = pd.read_excel(u"H:\\work\\usgc\\2015\\quant\\hgcdev0311g_shm.xlsx", sheetname = "data")
varhgcdev = [x.lower() for x in list(hgcdev_all)]
hgcdev_all.columns = varhgcdev

hgcdev = hgcdev_all.query('seg2 > 1')

# the following is not good, because it includes Non-Profit / Agri / Govn
# good ones: proportion by sector_group
# hgcdev.query('default_flag == "N"').sector_group.value_counts() / len(hgcdev.query('default_flag == "N"').sector_group)
# hgcdev.query('default_flag == "Y"').sector_group.value_counts() / len(hgcdev.query('default_flag == "Y"').sector_group)
# sum(hgcdev.default_flag == 'Y')  		# 174
# sum(hgcdev.default_flag == 'N')		# 898

#################################  prepare calc for stratified sampling, excluding AGRI and NON-Profit  ##############################
# overall proportion by sector_group
hgcdev.sector_group.value_counts() / len(hgcdev.sector_group)

hgcdev_good_sector = hgcdev.query('default_flag == "N" & sector_group != "AGRI" & sector_group != "NONP"')
hgcdev_bad_sector = hgcdev.query('default_flag == "Y" & sector_group != "AGRI" & sector_group != "NONP"')

hgcdev_good_sector.shape        # 804 Good
hgcdev_bad_sector.shape			# 168 Bad

hgcdev_good_sector.sector_group.value_counts() 
# SRVC    245
# MFG     148
# WHLS    121
# CONS     88
# RETL     85
# REAL     65
# TRAN     32
# FIN      15
# GOVT      3
# MINE      2

# F2012 has 58 defaults, so need 58 * 804 / 168 = 278 goods  (after SIC, there is only 45 defaults, so need 45 * 804 / 168 = 220 goods)  
# F2013 has 45 defaults, so need 45 * 804 / 168 = 216 goods
# F2014 has 49 defaults, so need 49 * 804 / 168 = 235 goods

good_num_2012 = np.floor(45 * hgcdev_good_sector.sector_group.value_counts() / len(hgcdev_bad_sector.sector_group)) + 1      	#confirmed
good_num_2013 = np.floor(29 * hgcdev_good_sector.sector_group.value_counts() / len(hgcdev_bad_sector.sector_group)) + 1 		#
good_num_2014 = np.floor(42 * hgcdev_good_sector.sector_group.value_counts() / len(hgcdev_bad_sector.sector_group)) + 1			#














