import pandas as pd
import numpy as np
import Tkinter

# read in loan info
loan_info = pd.read_csv( r'C:\Users\shm\Downloads\maturity_lifetime_el_engine.csv')

# part 1: prep for 0 to 75 months
cum_pd = pd.read_excel(r'C:\Users\shm\Downloads\Effective Maturity Tables & Lifetime EL Calculation.xlsx', sheetname = 'cum_pd')
cum_pd.columns = [x.lower() for x in cum_pd.columns]
cum_pd['month_i'] = range(3, (cum_pd.shape[0] + 1)* 3, 3)

row0 = pd.DataFrame([0, 0]).T
row0.columns = cum_pd.columns
cum_pd = pd.concat([row0, cum_pd], axis = 0)

cum_pd['month_incr'] = cum_pd.cumulative_pd.diff().fillna(0) / 3

# from 78 to 360
cum_after = pd.DataFrame(columns = cum_pd.columns)
cum_after['month_i'] = np.arange(cum_pd['month_i'].max() + 3, 360 + 3, 3)		# need to change 360?
cum_after['month_incr'] = cum_pd.month_incr.values[-1]

cum_after.ix[0, 'cumulative_pd'] = cum_pd.cumulative_pd.values[-1] * 2 - cum_pd.cumulative_pd.values[-2] 
for i in range(1, cum_after.shape[0]):
	cum_after.ix[i, 'cumulative_pd'] = cum_after.ix[i - 1, 'cumulative_pd'] + cum_pd.cumulative_pd.values[-1] - cum_pd.cumulative_pd.values[-2] 

final_cum_pd = pd.concat([cum_pd, cum_after], ignore_index = True)

# part 2: create default_rate curve
def_rate_curve = pd.DataFrame(columns = ['month_i', 'rate'])
def_rate_curve.month_i = np.arange(1, 360 + 1)		# need to change 360?
def_rate_curve.rate = final_cum_pd.ix[np.digitize(def_rate_curve.month_i, final_cum_pd.month_i) - 1, 'month_incr'].values

 
# part 3: calculate sheet  cashflow calculation engine  for each loan
def calc_engine(original_balance, term_month, yr_int_rate, month_prepay, lgd, month_drawdown_rate):

	remain_term_in_month = term_month
	current_balance = original_balance
	payment = -np.pmt(yr_int_rate / 12, term_month, original_balance)

	common_columns = ['monthIndex', 'starting_balance', 'schedule_payment', 'interest_paid', 'amortized_amount', 'prepay_amount', 'default_amount', 'recovery_amount', 'loss_amount', 'cash_drawdown_amount', 'ending_balance', 'total_cashflow', 'discount_cashflow', 'negative_balance', 'discount_loss']
	detailed_calc_result = pd.DataFrame(index = range(1, int(term_month) + 1), columns = common_columns)

	# to calculate for each column in excel file	
	for monthIndex in range(1, int(term_month) + 1):
		if monthIndex == 1:
			# D: starting_balance
			starting_balance = current_balance
			# E: schedule_payment
			if monthIndex >= remain_term_in_month:
				schedule_payment = starting_balance * (1 + yr_int_rate / 12)
			else:
				if starting_balance < payment:
					schedule_payment = starting_balance
				else:
					schedule_payment = payment
			# F: interest_paid    G: amortized_amount
			interest_paid = starting_balance * yr_int_rate / 12
			amortized_amount = schedule_payment - interest_paid
			# H: prepay_amount
			if ((monthIndex >= remain_term_in_month) | (starting_balance <= schedule_payment)):
				prepay_amount = 0
			else:
				prepay_amount = starting_balance * month_prepay
			# I: default_amount
			default_amount = (starting_balance - prepay_amount) * def_rate_curve.ix[def_rate_curve.month_i == monthIndex, 'rate'].values[0]
			# J: recovery_amount     K: loss_amount     L: cash_drawdown_amount
			recovery_amount = default_amount * (1 - lgd)
			loss_amount = default_amount - recovery_amount
			cash_drawdown_amount = original_balance * month_drawdown_rate
			# M: ending_balance     N: total_cashflow
			ending_balance = starting_balance - amortized_amount - prepay_amount - default_amount + cash_drawdown_amount
			total_cashflow = schedule_payment + prepay_amount - loss_amount - cash_drawdown_amount
			# O: discount_cashflow     P: negative_balance
			discount_cashflow = total_cashflow / ((1 + yr_int_rate / 12) ** monthIndex)
			# P: negative_balance
			if ((ending_balance <= 0.1) | (monthIndex >= remain_term_in_month)):
				negative_balance = 1
			else:
				negative_balance = 0
			# R: discount_loss
			discount_loss = loss_amount / ((1 + yr_int_rate / 12) ** monthIndex)
			detailed_calc_result.ix[monthIndex, :] = [monthIndex, starting_balance, schedule_payment, interest_paid, amortized_amount, prepay_amount, default_amount, recovery_amount, loss_amount, cash_drawdown_amount, ending_balance, total_cashflow, discount_cashflow, negative_balance, discount_loss]
		else:
			# D: starting_balance
			starting_balance = ending_balance
			# E: schedule_payment
			if monthIndex >= remain_term_in_month:
				schedule_payment = starting_balance * (1 + yr_int_rate / 12)
			else:
				if starting_balance < payment:
					schedule_payment = starting_balance
				else:
					schedule_payment = payment
			# F: interest_paid    G: amortized_amount
			interest_paid = starting_balance * yr_int_rate / 12
			amortized_amount = schedule_payment - interest_paid
			# H: prepay_amount
			if ((monthIndex >= remain_term_in_month) | (starting_balance <= schedule_payment)):
				prepay_amount = 0
			else:
				prepay_amount = starting_balance * month_prepay
			# I: default_amount
			default_amount = (starting_balance - prepay_amount) * def_rate_curve.ix[def_rate_curve.month_i == monthIndex, 'rate'].values[0]
			# J: recovery_amount     K: loss_amount     L: cash_drawdown_amount
			recovery_amount = default_amount * (1 - lgd)
			loss_amount = default_amount - recovery_amount
			cash_drawdown_amount = original_balance * month_drawdown_rate
			# M: ending_balance     N: total_cashflow
			ending_balance = starting_balance - amortized_amount - prepay_amount - default_amount + cash_drawdown_amount
			total_cashflow = schedule_payment + prepay_amount - loss_amount - cash_drawdown_amount
			# O: discount_cashflow     P: negative_balance
			discount_cashflow = total_cashflow / ((1 + yr_int_rate / 12) ** monthIndex)
			# P: negative_balance
			if ((ending_balance <= 0.1) | (monthIndex >= remain_term_in_month)):
				negative_balance = 1
			else:
				negative_balance = 0
			# R: discount_loss
			discount_loss = loss_amount / ((1 + yr_int_rate / 12) ** monthIndex)
			detailed_calc_result.ix[monthIndex, :] = [monthIndex, starting_balance, schedule_payment, interest_paid, amortized_amount, prepay_amount, default_amount, recovery_amount, loss_amount, cash_drawdown_amount, ending_balance, total_cashflow, discount_cashflow, negative_balance, discount_loss]
		
	# calculate with detailed method	
	ending_date = detailed_calc_result.query('negative_balance == 1').monthIndex.values[0]	
	selected_data = detailed_calc_result.ix[detailed_calc_result.monthIndex <= ending_date, :]
	life_time_value = selected_data['discount_cashflow'].sum()
	effective_life = np.sum(selected_data.discount_cashflow * selected_data.monthIndex)	/ np.sum(selected_data.discount_cashflow)
	total_loss_w_discount = np.sum(selected_data.discount_loss)

	# calculate by prorate pd
	lifetime_pd_lookup = final_cum_pd.ix[np.digitize([effective_life], final_cum_pd.month_i) - 1, 'cumulative_pd'].values[0]
	ead = original_balance
	lifetime_el = lifetime_pd_lookup * lgd * ead
	discount_at_half_effective_life = lifetime_el /((1 + yr_int_rate / 12) ** (effective_life / 2))
	
	input_summary = pd.DataFrame([original_balance, term_month, yr_int_rate, month_prepay, lgd, month_drawdown_rate]).T
	input_summary.columns = ['original_balance', ' term_month', ' yr_int_rate', ' month_prepay', ' lgd', ' month_drawdown_rate']
	
	output_compare = pd.DataFrame([ending_date, life_time_value, effective_life, total_loss_w_discount, lifetime_pd_lookup, lifetime_el, discount_at_half_effective_life]).T	
	output_compare.columns = ['ending_date', ' life_time_value', ' effective_life', ' total_loss_w_discount', ' lifetime_pd_lookup', ' lifetime_el', ' discount_at_half_effective_life']

	output_sumary = pd.concat([input_summary, output_compare], axis = 1)	
	return output_sumary	
	
result = pd.DataFrame()	
for each_loan in loan_info.loan_id:
	vars = loan_info.ix[loan_info.loan_id == each_loan, 1:].values[0]
	s = calc_engine(*vars)
	loan_result = pd.concat([pd.DataFrame([each_loan], columns = ['loan_id']), s], axis = 1)
	result = pd.concat([loan_result, result], ignore_index = 1)
	
print result	
	
	
	
	
	
	
	
	
	
	










