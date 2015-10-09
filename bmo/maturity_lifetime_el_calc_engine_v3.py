## v3 update 20151005: update to calculate for different rating. v3 is from v2

import pandas as pd
import numpy as np
import Tkinter

# read in loan info
loan_info = pd.read_excel( r'H:\work\IFRS9\PCUS\CI\maturity_lifetime_el_engine_input.xlsx', sheetname = 'loan_input_example')

# default_rate_curve for '200803' for each rating will be calculated in ifrs9_pcus_cni_combine.py 

 
# part 3: calculate sheet  cashflow calculation engine  for each loan
def calc_engine(original_balance, term_month, yr_int_rate, month_prepay, lgd, month_drawdown_rate, current_rr):

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
			default_amount = (starting_balance - prepay_amount) * default_curve_data[current_rr].ix[default_curve_data[current_rr].month_i == monthIndex, 'rate'].values[0]
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
			default_amount = (starting_balance - prepay_amount) * default_curve_data[current_rr].ix[default_curve_data[current_rr].month_i == monthIndex, 'rate'].values[0]
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
	vars = loan_info.ix[loan_info.loan_id == each_loan, 1:8].values[0]
	s = calc_engine(*vars)
	loan_result = pd.concat([pd.DataFrame([each_loan], columns = ['loan_id']), s], axis = 1)
	result = pd.concat([loan_result, result], ignore_index = 1)
	
print result	
	
result.to_excel(r'H:\work\IFRS9\Jimmy_calc_engine\calc_engine_output.xlsx')	
	
	
	
	
	
	
	
	










