
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import bisect

hgcdev_all = pd.read_excel(u"H:\\work\\usgc\\2015\\quant\\hgcdev0311g_shm.xlsx", sheetname = "data")
varhgcdev = [x.lower() for x in list(hgcdev_all)]
hgcdev_all.columns = varhgcdev

hgcdev = hgcdev_all.query('seg2 > 1')

# overall proportion by sector_group
hgcdev.sector_group.value_counts() / len(hgcdev.sector_group)
# good ones: proportion by sector_group
hgcdev.query('default_flag == "N"').sector_group.value_counts() / len(hgcdev.query('default_flag == "N"').sector_group)
hgcdev.query('default_flag == "Y"').sector_group.value_counts() / len(hgcdev.query('default_flag == "Y"').sector_group)
 
