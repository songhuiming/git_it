
#!/usr/bin/python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

mra_vars = ['sk_entity_id', 'as_of_dt', 'cur_ast_amt', 'tot_ast_amt', 'cur_liab_amt', 'tot_liab_amt', 'tot_debt_amt', 'net_worth_amt', 'tangible_net_worth_amt', 'tot_sales_amt', 'net_inc_amt', 'ebitda_amt', 'tot_debt_srvc_amt', 'ebit_amt', 'cash_and_secu_amt', 'debt_srvc_cov_rto', 'cur_rto', 'debt_to_tnw_rto', 'debt_to_ebitda_rto']

hbc_11_13 = pd.read_excel("R:\\Global_Market_Risk\RCST\\WCM\Huiming_Song\\data\\2014-03 Harris PD Validation Data F2013\\HBC_20110101_20131031.xlsx", sheetname = "HGC")
hbcvars = [x.replace(' ', '_').lower() for x in list(hbc_11_13)]
hbc_11_13.columns = hbcvars

mra = pd.read_excel("R:\\Global_Market_Risk\RCST\\WCM\Huiming_Song\\data\\DCU_MRA_RISK_SPREAD_D.xls", sheetname = "DCU_MRA_RISK_SPREAD_D")
mravars = [x.replace(' ', '_').lower() for x in list(mra)]
mra.columns = mravars
 
hbc_mra = pd.merge(hbc_11_13, mra, on = ['sk_entity_id', 'as_of_dt'], how = 'inner')


# relax match condition
df2 = pd.merge(bmo2013HBC_2_after_sic_pw, mra, left_on = ['sk_entity_id'], right_on = ['sk_entity_id'], how = 'inner')
df2['dt_diff'] = abs((df2.final_form_date - df2.as_of_dt) / np.timedelta64(1, 'D'))
df2['priority'] = np.where(df2['dt_diff'] == 0, 1, np.where(df2['dt_diff'] <= 30, 2, np.where(df2['dt_diff'] <= 50, 3, 4)))
df2.drop_duplicates('uen').priority.value_counts()
df2.drop_duplicates('uen').query('default_flag == 1').priority.value_counts()

df2_dudup = df2.drop_duplicates('uen')
