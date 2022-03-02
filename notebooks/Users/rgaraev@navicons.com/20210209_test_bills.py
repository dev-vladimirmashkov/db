# Databricks notebook source
import pandas as pd
import numpy as np

# plotlib
import seaborn as sns
import matplotlib.pyplot as plt

import datetime
from datetime import timedelta
import time
import os
import gc
from shutil import copyfile
from tqdm import tqdm
import pickle

import warnings
# warnings.filterwarnings('ignore')

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
from scipy.optimize import minimize

pd.set_option("display.max_columns", 500)
pd.set_option("display.max_rows", 500)

print(pd.__version__)

# COMMAND ----------

# функция WAPE
def wape(y_pred, y_true):
    res = np.sum(np.abs(y_true - y_pred)) / np.abs(np.sum(y_true)) * 100
    return res

# COMMAND ----------

YL = pd.read_csv('/dbfs/FileStore/shared_uploads/rgaraev@navicons.com/YL_data__1_')
YL.drop('Unnamed: 0',axis=1,inplace=True)
YL

# COMMAND ----------

YL['Price'] = YL.Money/YL.Q
YL

# COMMAND ----------

next0 = YL.groupby(['code'])['date', 'Q', 'Money', 'Brand', 'Product', 'Price'].last().reset_index()
next0

# COMMAND ----------

next1 = next0.copy()
next2 = next0.copy()
next3 = next0.copy()

# COMMAND ----------

next1['date'] = '2020.01'
next1['Q'] = np.nan
next1['Money'] = np.nan

next2['date'] = '2020.02'
next2['Q'] = np.nan
next2['Money'] = np.nan

next3['date'] = '2020.03'
next3['Q'] = np.nan
next3['Money'] = np.nan

YL = pd.concat([YL, next1, next2, next3], axis=0)

# COMMAND ----------

YL.date = YL.date.astype(str)

# COMMAND ----------

YL

# COMMAND ----------

YL['Date'] = pd.to_datetime(YL.date.astype(str), format='%Y.%m')
YL

# COMMAND ----------

# MAGIC %%time
# MAGIC for j in range(1,6):
# MAGIC     YL['SalesQ_lag'+str(j)] = YL.groupby(['code'])['Q'].transform(lambda x: x.shift(j))
# MAGIC     YL['SalesM_lag'+str(j)] = YL.groupby(['code'])['Money'].transform(lambda x: x.shift(j))
# MAGIC 
# MAGIC print('rollings')
# MAGIC for l in range(1,4):
# MAGIC     for r in [2,3,6,9]:
# MAGIC         YL['rolling_mean_Q'+str(l)+'_'+str(r)] = YL.groupby(['code'])['Q'].transform(lambda x: x.shift(l).rolling(r).mean())
# MAGIC         YL['rolling_std_Q'+str(l)+'_'+str(r)] = YL.groupby(['code'])['Q'].transform(lambda x: x.shift(l).rolling(r).std())
# MAGIC         YL['rolling_mean_Money'+str(l)+'_'+str(r)] = YL.groupby(['code'])['Money'].transform(lambda x: x.shift(l).rolling(r).mean())
# MAGIC         YL['rolling_std_Money'+str(l)+'_'+str(r)] = YL.groupby(['code'])['Money'].transform(lambda x: x.shift(l).rolling(r).std())

# COMMAND ----------

YL.columns

# COMMAND ----------

YL['CountBrand'] = YL.groupby('Brand')['code'].transform('count').astype(np.float16)
YL.sample(7)

# COMMAND ----------

YL['year'] = YL.Date.dt.strftime('%Y').astype(int)
YL['month'] = YL.Date.dt.strftime('%m').astype(int)
YL['y_m'] = YL.Date.dt.strftime('%Y%m').astype(int)
YL.head()

# COMMAND ----------

grid_df = YL[YL.y_m < 201907][['code', 'date', 'Q', 'Money', 'Brand', 'Product', 'Price', 'Date']]
TARGET  = 'Q'
col = 'Brand'
CatAll = grid_df.groupby(col).agg({
        'Q':['mean','std']}).reset_index()
CatAll.columns = [col,'enc'+col+'_mean','enc'+col+'_std']
YL = YL.merge(CatAll, how='left', on=col)

# COMMAND ----------

YL[['code', 'date', 'Q', 'Money', 'Brand', 'Product', 'encBrand_mean', 'encBrand_std','CountBrand']].sample(10)

# COMMAND ----------

YL['Price_lag'] = YL.groupby(['code'])['Price'].transform(lambda x: x.shift(1))
YL['Price_change'] = YL.Price / YL.Price_lag

# COMMAND ----------

YL[['code', 'date', 'Q', 'Money', 'Brand', 'Product', 'Price', 'Price_lag','Price_change']].sample(10)

# COMMAND ----------

YL[YL.Price_change < 1].head()

# COMMAND ----------

YL['Price_lag'] = YL.groupby(['code'])['Price'].transform(lambda x: x.shift(-1))
YL['Price_grow'] = YL.Price / YL.Price_lag
YL[YL.Price_grow < 1].head()

# COMMAND ----------

YL.tail()

# COMMAND ----------

YL.loc[YL.Price_grow.isna(), 'Price_grow'] = 1

# COMMAND ----------

YL.tail()

# COMMAND ----------

YL.describe()

# COMMAND ----------

YL[YL.code == 1602]

# COMMAND ----------

YL.rolling_mean_Money1_2

# COMMAND ----------

YL.columns

# COMMAND ----------

import lightgbm as lgb
lgb_params = {
                    'boosting_type': 'gbdt',
                    'objective': 'regression',
                    'metric': 'l2',
                    'num_leaves': 31,
                    'learning_rate': 0.05,
                    'feature_fraction': 0.9,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'verbose': 0
                } 

# COMMAND ----------

TARGET  = 'Q' 
features = ['code',
 'rolling_mean_Money3_6',
 'encBrand_std',
 'Price_change',
 'CountBrand',
 'encBrand_mean',
 'rolling_mean_Q3_3',
 'SalesQ_lag3',
 'rolling_std_Money3_3',
 'SalesQ_lag5',
 'SalesM_lag5',
 'SalesM_lag3',
 'rolling_std_Q3_2',
 'SalesM_lag4',
 'rolling_mean_Money3_2',
 'Price_grow',
 'Price_lag',
 'rolling_std_Q3_3',
 'rolling_std_Money3_6',
 'rolling_mean_Q3_6',
 'month',
 'Price',
 'rolling_mean_Q3_2',
 'year',
 'rolling_std_Money3_2',
 'rolling_std_Q3_6',
 'rolling_mean_Money3_3']

# COMMAND ----------

train = lgb.Dataset(YL[YL.y_m<201909][features],
                   YL[YL.y_m<201909][TARGET])

# COMMAND ----------

test = lgb.Dataset(YL[YL.y_m>=201909][features],
                   YL[YL.y_m>=201909][TARGET])

# COMMAND ----------

model = lgb.train(lgb_params,
                 train,
                 num_boost_round=200)

# COMMAND ----------

preds = model.predict(YL[YL.y_m>=201909][features])

# COMMAND ----------

wape(preds,YL[YL.y_m>=201909][TARGET])

# COMMAND ----------

preds_2019_09 = model.predict(YL[YL.y_m>=201909][features])

# COMMAND ----------

preds = pd.DataFrame(preds_2019_09,columns=['preds_for_2019_09_and_more'])

# COMMAND ----------

