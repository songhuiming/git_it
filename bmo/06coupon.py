
## project details:  https://www.kaggle.com/c/coupon-purchase-prediction

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

import statsmodels.formula.api as smf

# coupon_list_train.csv - the master list of coupons which are considered part of the training set
listtrain = pd.read_csv(r'H:\python\kaggle\06coupons\coupon_list_train.csv')  # coupon_id_hash level
listtrain.columns = [x.lower() for x in listtrain.columns]

# coupon_area_train.csv - the coupon listing area for the training set coupons
areatrain = pd.read_csv(r'H:\python\kaggle\06coupons\coupon_area_train.csv')
areatrain.columns = [x.lower() for x in areatrain.columns]

# coupon_detail_train.csv - the purchase log of users buying coupons during the training set time period. You are not provided this table for the test set period.
detailtrain = pd.read_csv(r'H:\python\kaggle\06coupons\coupon_detail_train.csv') # purchaseid level
detailtrain.columns = [x.lower() for x in detailtrain.columns]

# coupon_visit_train.csv - the viewing log of users browsing coupons during the training set time period. You are not provided this table for the test set period.
visittrain = pd.read_csv(r'H:\python\kaggle\06coupons\coupon_visit_train.csv')
visittrain.columns = [x.lower() for x in visittrain.columns]

capsule = pd.read_excel(r'H:\python\kaggle\06coupons\jpn_2_english.xlsx', sheetname = 'capsule')
capsule.columns = [x.lower() for x in capsule.columns]

genre = pd.read_excel(r'H:\python\kaggle\06coupons\jpn_2_english.xlsx', sheetname = 'genre')
genre.columns = [x.lower() for x in genre.columns]

# user list file
userlist = pd.read_csv(r'H:\python\kaggle\06coupons\user_list.csv')
userlist.columns = [x.lower() for x in userlist.columns]

# merge visit table with coupon info
train_data = pd.merge(listtrain, visittrain, left_on = 'coupon_id_hash', right_on = 'view_coupon_id_hash', how = 'inner')

f = 'purchase_flg ~ C(genre_name) + price_rate + discount_price + C(usable_date_mon)'
logfit = smf.logit(formula = str(f), data = train_data).fit()
