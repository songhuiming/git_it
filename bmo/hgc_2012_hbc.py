
#!/usr/bin/python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
 
#read in the 2012 HBC ratings
bmo2012HBC = pd.read_excel(u"H:\\work\\usgc\\2015\\quant\\2012\\2012_combined_data.xlsx", sheetname = "HBC_201111_201201")
bmo2012HBC['source_from'] = 'HBC'
varbmo2012HBC = [x.replace(' ', '_').lower() for x in list(bmo2012HBC)]
bmo2012HBC.columns = varbmo2012HBC
bmo2012HBC_1 = bmo2012HBC.ix[bmo2012HBC.rnd != 'Y', ['sk_entity_id', 'entity_nm', 'eff_dt', 'fin_stmt_dt', 'source_from', 'input_siccd']]   
bmo2012HBC_1 = bmo2012HBC_1.rename(columns = {'entity_nm': 'entity_name', 'eff_dt': 'final_form_date', 'input_siccd': 'us_sic'})


#merge with population to get default flag
hgcdf = pd.read_excel(u"H:\\work\\usgc\\2015\\quant\\2012\\2012_combined_data.xlsx", sheetname = "populationWithDefault")
hgcdf.columns = [x.replace(' ', '_').lower() for x in list(hgcdf)]
hgcdf[hgcdf.duplicated([u'bor_sk'])]  #duplicated of 21094192
#de-dup population data for defaults
hgcdf.drop_duplicates(['bor_sk'], inplace = True)
hgcdf = hgcdf.rename(columns = {'bor_sk': 'sk_entity_id'})
hgcdf['default_flag'] = hgcdf.default_flag.replace({'Y': 1, 'N':0})

# read in the sic_indust data
sic_indust = pd.read_excel("H:\\work\\usgc\\2015\\quant\\SIC_Code_List.xlsx", sheetname = "sic_indust") 
 
#join 
bmo2012HBC_2 = pd.merge(bmo2012HBC_1, hgcdf, on = "sk_entity_id", how = "inner")
bmo2012HBC_2.default_flag.value_counts(dropna = False).sort_index()        		# {0: 570, 1: 38, nan: 125}, need to change to inner join
 

bmo2012HBC_2['insudt'] = bmo2012HBC_2.us_sic.replace(dict(zip(sic_indust.sic_code, sic_indust.indust)))
bmo2012HBC_2['sector_group'] = bmo2012HBC_2.us_sic.replace(dict(zip(sic_indust.sic_code, sic_indust.sector_group)))    #3251, 7312 cannot be matched

outfile_12 = u"H:\\work\\usgc\\2015\\quant\\2012\\hgc2012_output_after_pw_" + pd.datetime.today().strftime("%Y_%m_%d") + '.xlsx'
writer = pd.ExcelWriter(outfile_12) 
bmo2012HBC_2.to_excel(writer, sheet_name = "bmo2012HBC_2")

# remove the not included industry sectors
bmo2012HBC_after_sic = bmo2012HBC_2.query('sector_group in ["SRVC", "MFG", "WHLS", "CONS", "RETL", "TRAN", "FIN", "REAL", "GOVT", "MINE"]')  

#sort for dedup to sk_entity_id level
bmo2012HBC_after_sic.sort(['sk_entity_id', 'final_form_date'], inplace = True)
bmo2012HBC_after_sic_pw = bmo2012HBC_after_sic.drop_duplicates('sk_entity_id')     					# 372 in total, {0: 350, 1: 22}
bmo2012HBC_after_sic_pw.to_excel(writer, sheet_name = "bmo2012HBC_after_sic")

writer.save()

print bmo2012HBC_after_sic_pw.sector_group.value_counts(dropna = False)
print bmo2012HBC_after_sic_pw.default_flag.value_counts(dropna = False)   		 
 



