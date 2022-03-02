# Databricks notebook source
import pandas as pd
import numpy as np

# plotlib
import seaborn as sns
import matplotlib.pyplot as plt

import datetime
from datetime import timedelta
import time
import pyarrow
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
print(pyarrow.__version__)

# COMMAND ----------

# Блокнот выполняет feature_engineering и отбор признаков на продуктовых данных
# Блокнот отработал за полчаса
# 68 продуктов, 18 брендов
# 1 Driver: 14.0 GB Memory, 4 Cores, 0.75 DBU, 0 workers

# COMMAND ----------

# функция WAPE
def wape(y_pred, y_true):
    res = np.sum(np.abs(y_true - y_pred)) / np.abs(np.sum(y_true)) * 100
    return res

# COMMAND ----------

YL = pd.read_csv('/dbfs/FileStore/shared_uploads/rgaraev@navicons.com/YL_data__1_')
YL.drop('Unnamed: 0',axis=1,inplace=True)

# COMMAND ----------

YL

# COMMAND ----------

YL[YL.Q == YL.Money]

# COMMAND ----------

YL['Price'] = YL.Money/YL.Q
YL

# COMMAND ----------

YL.describe()


# COMMAND ----------

YL.columns

# COMMAND ----------

YL.code.unique()

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

train_df = YL[YL['y_m'] > 201706].reset_index(drop=True)
train_df.head()

# COMMAND ----------

graf = YL.groupby(['y_m'])['year'].first().reset_index()
base = graf['y_m'].reset_index(drop=True).reset_index()
base.columns = ['wm_yr_wk','y_m']
base.head()

# COMMAND ----------

base.iloc[-1-4,1]

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

def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:42].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (averaged predictions)')
    plt.tight_layout()

# COMMAND ----------

TARGET  = 'Q' 
features1 = ['code',  'Price', 
       'SalesQ_lag1', 'SalesM_lag1', 'SalesQ_lag2', 'SalesM_lag2',
       'SalesQ_lag3', 'SalesM_lag3', 'SalesQ_lag4', 'SalesM_lag4',
       'SalesQ_lag5', 'SalesM_lag5', 'rolling_mean_Q1_2', 'rolling_std_Q1_2',
       'rolling_mean_Money1_2', 'rolling_std_Money1_2', 'rolling_mean_Q1_3',
       'rolling_std_Q1_3', 'rolling_mean_Money1_3', 'rolling_std_Money1_3',
       'rolling_mean_Q1_6', 'rolling_std_Q1_6', 'rolling_mean_Money1_6',
       'rolling_std_Money1_6', 
#              'rolling_mean_Q1_9', 'rolling_std_Q1_9',
#        'rolling_mean_Money1_9', 'rolling_std_Money1_9', 
#              'rolling_mean_Q2_2',
#        'rolling_std_Q2_2', 'rolling_mean_Money2_2', 'rolling_std_Money2_2',
#        'rolling_mean_Q2_3', 'rolling_std_Q2_3', 'rolling_mean_Money2_3',
#        'rolling_std_Money2_3', 'rolling_mean_Q2_6', 'rolling_std_Q2_6',
#        'rolling_mean_Money2_6', 'rolling_std_Money2_6', 'rolling_mean_Q2_9',
#        'rolling_std_Q2_9', 'rolling_mean_Money2_9', 'rolling_std_Money2_9',
#        'rolling_mean_Q3_2', 'rolling_std_Q3_2', 'rolling_mean_Money3_2',
#        'rolling_std_Money3_2', 'rolling_mean_Q3_3', 'rolling_std_Q3_3',
#        'rolling_mean_Money3_3', 'rolling_std_Money3_3', 'rolling_mean_Q3_6',
#        'rolling_std_Q3_6', 'rolling_mean_Money3_6', 'rolling_std_Money3_6',
#        'rolling_mean_Q3_9', 'rolling_std_Q3_9', 'rolling_mean_Money3_9',
#        'rolling_std_Money3_9', 
             'CountBrand', 'year', 'month', 'y_m',
       'encBrand_mean', 'encBrand_std', 'Price_lag', 'Price_change',
       'Price_grow']

# COMMAND ----------

# MAGIC %%time
# MAGIC rest_cols = features1
# MAGIC WAPEmean = np.zeros(len(rest_cols))
# MAGIC NumIter = np.zeros(len(rest_cols))
# MAGIC q = 0
# MAGIC used_cols = []
# MAGIC while len(rest_cols) > 0:
# MAGIC     best_col = ''
# MAGIC     best_col_error = 1e6
# MAGIC     for col in rest_cols:
# MAGIC         features = used_cols + [col]
# MAGIC         # LightGBM тест ООF на +1
# MAGIC         errors = []
# MAGIC         num=200
# MAGIC         for i in range(1,6):
# MAGIC             test_m = 201912
# MAGIC             test = base.iloc[-i-4,1]
# MAGIC             test1 = base.iloc[-i-3,1]
# MAGIC             print(test)
# MAGIC             train_data = lgb.Dataset(train_df[train_df.y_m < test][features], 
# MAGIC                                    label=train_df[train_df.y_m < test][TARGET])
# MAGIC             valid_data = lgb.Dataset(train_df[train_df.y_m == test][features], 
# MAGIC                                    label=train_df[train_df.y_m == test][TARGET])
# MAGIC             estimator = lgb.train(lgb_params,
# MAGIC                                   train_data,
# MAGIC                                   valid_sets = [valid_data],
# MAGIC                                   num_boost_round=num,
# MAGIC                                   early_stopping_rounds=40,
# MAGIC                                 verbose_eval=-1                             
# MAGIC                                  )
# MAGIC 
# MAGIC             error = wape(estimator.predict(train_df[train_df.y_m == test1][features]), train_df.loc[train_df.y_m == test1, TARGET])
# MAGIC             errors.append(error)    
# MAGIC        
# MAGIC         col_error = np.mean(np.array(errors))
# MAGIC         if col_error < best_col_error:
# MAGIC             best_col = col
# MAGIC             best_col_error = col_error
# MAGIC     print(features)    
# MAGIC     print('Лучшая средняя ошибка M1:')
# MAGIC     print(best_col_error)
# MAGIC     WAPEmean[q] = np.mean(np.array(errors))
# MAGIC 
# MAGIC     used_cols.append(best_col)
# MAGIC     rest_cols = [x for x in rest_cols if x != best_col]
# MAGIC     q += 1        
# MAGIC 
# MAGIC     

# COMMAND ----------

i_min = np.argmin(WAPEmean)
print(i_min)
print(WAPEmean[i_min])



# COMMAND ----------

# MAGIC %%time
# MAGIC rest_cols = features1
# MAGIC WAPEmean = np.zeros(len(rest_cols))
# MAGIC NumIter = np.zeros(len(rest_cols))
# MAGIC q = 0
# MAGIC used_cols = []
# MAGIC while q <= i_min:
# MAGIC     best_col = ''
# MAGIC     best_col_error = 1e6
# MAGIC     for col in rest_cols:
# MAGIC         features = used_cols + [col]
# MAGIC         # LightGBM тест ООF на +1
# MAGIC         errors = []
# MAGIC         num=200
# MAGIC         for i in range(1,6):
# MAGIC             test_m = 201912
# MAGIC             test = base.iloc[-i-4,1]
# MAGIC             test1 = base.iloc[-i-3,1]
# MAGIC             print(test)
# MAGIC             train_data = lgb.Dataset(train_df[train_df.y_m < test][features], 
# MAGIC                                    label=train_df[train_df.y_m < test][TARGET])
# MAGIC             valid_data = lgb.Dataset(train_df[train_df.y_m == test][features], 
# MAGIC                                    label=train_df[train_df.y_m == test][TARGET])
# MAGIC             estimator = lgb.train(lgb_params,
# MAGIC                                   train_data,
# MAGIC                                   valid_sets = [valid_data],
# MAGIC                                   num_boost_round=num,
# MAGIC                                   early_stopping_rounds=40,
# MAGIC                                 verbose_eval=-1                             
# MAGIC                                  )
# MAGIC 
# MAGIC             error = wape(estimator.predict(train_df[train_df.y_m == test1][features]), train_df.loc[train_df.y_m == test1, TARGET])
# MAGIC             errors.append(error)    
# MAGIC        
# MAGIC         col_error = np.mean(np.array(errors))
# MAGIC         if col_error < best_col_error:
# MAGIC             best_col = col
# MAGIC             best_col_error = col_error
# MAGIC     print(features)    
# MAGIC     print('Лучшая средняя ошибка M1:')
# MAGIC     print(best_col_error)
# MAGIC     WAPEmean[q] = np.mean(np.array(errors))
# MAGIC 
# MAGIC     used_cols.append(best_col)
# MAGIC     rest_cols = [x for x in rest_cols if x != best_col]
# MAGIC     q += 1    

# COMMAND ----------

used_cols

# COMMAND ----------

# MAGIC %%time
# MAGIC WAPEmean = np.zeros(len(range(25,501,25)))
# MAGIC NumIter = np.zeros(len(range(25,501,25)))
# MAGIC q = 0
# MAGIC for n in range(25,501,25):
# MAGIC     # LightGBM тест ООF на +1
# MAGIC     errors = []
# MAGIC     train_df['OOF'] = 0
# MAGIC     feature_importance_df = pd.DataFrame()
# MAGIC     print(n)
# MAGIC     num=n
# MAGIC     YL['OOF1a'] = 0
# MAGIC     YL['OOF2a'] = 0
# MAGIC     YL['OOF3a'] = 0
# MAGIC     for i in range(1,6):
# MAGIC         print('M1')
# MAGIC         features = used_cols
# MAGIC         test_m = 201912
# MAGIC         test = base.iloc[-i-4,1]
# MAGIC         test1 = base.iloc[-i-3,1]
# MAGIC         print(test)
# MAGIC         train_data = lgb.Dataset(train_df[train_df.y_m < test][features], 
# MAGIC                                label=train_df[train_df.y_m < test][TARGET])
# MAGIC         valid_data = lgb.Dataset(train_df[train_df.y_m == test][features], 
# MAGIC                                label=train_df[train_df.y_m == test][TARGET])
# MAGIC         estimator = lgb.train(lgb_params,
# MAGIC                               train_data,
# MAGIC                               valid_sets = [valid_data],
# MAGIC                               num_boost_round=num,
# MAGIC                               early_stopping_rounds=n/5,
# MAGIC                             verbose_eval=1000                             
# MAGIC                              )
# MAGIC 
# MAGIC         YL['OOF1a'] += estimator.predict(YL[features]) / 5
# MAGIC         error = wape(estimator.predict(train_df[train_df.y_m == test1][features]), train_df.loc[train_df.y_m == test1, TARGET])
# MAGIC         errors.append(error)    
# MAGIC #         print(error)
# MAGIC         importance_df = pd.DataFrame()
# MAGIC         importance_df["feature"] = features
# MAGIC         importance_df["importance"] = estimator.feature_importance()
# MAGIC         feature_importance_df = pd.concat([feature_importance_df, importance_df], axis=0)
# MAGIC 
# MAGIC     print('Средняя ошибка M1:')
# MAGIC     print(np.mean(np.array(errors)))
# MAGIC     WAPEmean[q] = np.mean(np.array(errors))
# MAGIC     NumIter[q] = n
# MAGIC     q += 1
# MAGIC i_min = np.argmin(WAPEmean)
# MAGIC print(i_min)
# MAGIC print(WAPEmean[i_min])
# MAGIC print(NumIter[i_min])

# COMMAND ----------

display_importances(feature_importance_df)

# COMMAND ----------

features1 = used_cols
num1 = NumIter[i_min]

# COMMAND ----------

TARGET  = 'Q' 
features2 = ['code',  'Price', 
#        'SalesQ_lag1', 
#              'SalesM_lag1',
             'SalesQ_lag2', 'SalesM_lag2',
       'SalesQ_lag3', 'SalesM_lag3', 'SalesQ_lag4', 'SalesM_lag4',
       'SalesQ_lag5', 'SalesM_lag5', 
#              'rolling_mean_Q1_2', 'rolling_std_Q1_2',
#        'rolling_mean_Money1_2', 'rolling_std_Money1_2', 'rolling_mean_Q1_3',
#        'rolling_std_Q1_3', 'rolling_mean_Money1_3', 'rolling_std_Money1_3',
#        'rolling_mean_Q1_6', 'rolling_std_Q1_6', 'rolling_mean_Money1_6',
#        'rolling_std_Money1_6', 
#              'rolling_mean_Q1_9', 'rolling_std_Q1_9',
#        'rolling_mean_Money1_9', 'rolling_std_Money1_9', 
             'rolling_mean_Q2_2',
       'rolling_std_Q2_2', 'rolling_mean_Money2_2', 'rolling_std_Money2_2',
       'rolling_mean_Q2_3', 'rolling_std_Q2_3', 'rolling_mean_Money2_3',
       'rolling_std_Money2_3', 'rolling_mean_Q2_6', 'rolling_std_Q2_6',
       'rolling_mean_Money2_6', 'rolling_std_Money2_6', 
#              'rolling_mean_Q2_9',
#        'rolling_std_Q2_9', 'rolling_mean_Money2_9', 'rolling_std_Money2_9',
#        'rolling_mean_Q3_2', 'rolling_std_Q3_2', 'rolling_mean_Money3_2',
#        'rolling_std_Money3_2', 'rolling_mean_Q3_3', 'rolling_std_Q3_3',
#        'rolling_mean_Money3_3', 'rolling_std_Money3_3', 'rolling_mean_Q3_6',
#        'rolling_std_Q3_6', 'rolling_mean_Money3_6', 'rolling_std_Money3_6',
#        'rolling_mean_Q3_9', 'rolling_std_Q3_9', 'rolling_mean_Money3_9',
#        'rolling_std_Money3_9', 
             'CountBrand', 'year', 'month', 'y_m',
       'encBrand_mean', 'encBrand_std', 'Price_lag', 'Price_change',
       'Price_grow']

# COMMAND ----------

# MAGIC %%time
# MAGIC rest_cols = features2
# MAGIC WAPEmean = np.zeros(len(rest_cols))
# MAGIC NumIter = np.zeros(len(rest_cols))
# MAGIC q = 0
# MAGIC used_cols = []
# MAGIC while len(rest_cols) > 0:
# MAGIC     best_col = ''
# MAGIC     best_col_error = 1e6
# MAGIC     for col in rest_cols:
# MAGIC         features = used_cols + [col]
# MAGIC         # LightGBM тест ООF на +1
# MAGIC         errors = []
# MAGIC         num=200
# MAGIC         for i in range(1,6):
# MAGIC             test_m = 201912
# MAGIC             test = base.iloc[-i-4,1]
# MAGIC             test1 = base.iloc[-i-3,1]
# MAGIC             print(test)
# MAGIC             train_data = lgb.Dataset(train_df[train_df.y_m < test][features], 
# MAGIC                                    label=train_df[train_df.y_m < test][TARGET])
# MAGIC             valid_data = lgb.Dataset(train_df[train_df.y_m == test][features], 
# MAGIC                                    label=train_df[train_df.y_m == test][TARGET])
# MAGIC             estimator = lgb.train(lgb_params,
# MAGIC                                   train_data,
# MAGIC                                   valid_sets = [valid_data],
# MAGIC                                   num_boost_round=num,
# MAGIC                                   early_stopping_rounds=40,
# MAGIC                                 verbose_eval=-1                             
# MAGIC                                  )
# MAGIC 
# MAGIC             error = wape(estimator.predict(train_df[train_df.y_m == test1][features]), train_df.loc[train_df.y_m == test1, TARGET])
# MAGIC             errors.append(error)    
# MAGIC        
# MAGIC         col_error = np.mean(np.array(errors))
# MAGIC         if col_error < best_col_error:
# MAGIC             best_col = col
# MAGIC             best_col_error = col_error
# MAGIC     print(features)    
# MAGIC     print('Лучшая средняя ошибка M2:')
# MAGIC     print(best_col_error)
# MAGIC     WAPEmean[q] = np.mean(np.array(errors))
# MAGIC 
# MAGIC     used_cols.append(best_col)
# MAGIC     rest_cols = [x for x in rest_cols if x != best_col]
# MAGIC     q += 1

# COMMAND ----------

i_min = np.argmin(WAPEmean)
print(i_min)
print(WAPEmean[i_min])


# COMMAND ----------

# MAGIC %%time
# MAGIC rest_cols = features2
# MAGIC WAPEmean = np.zeros(len(rest_cols))
# MAGIC NumIter = np.zeros(len(rest_cols))
# MAGIC q = 0
# MAGIC used_cols = []
# MAGIC while q < i_min + 1:
# MAGIC     best_col = ''
# MAGIC     best_col_error = 1e6
# MAGIC     for col in rest_cols:
# MAGIC         features = used_cols + [col]
# MAGIC         # LightGBM тест ООF на +1
# MAGIC         errors = []
# MAGIC         num=200
# MAGIC         for i in range(1,6):
# MAGIC             test_m = 201912
# MAGIC             test = base.iloc[-i-4,1]
# MAGIC             test1 = base.iloc[-i-3,1]
# MAGIC             print(test)
# MAGIC             train_data = lgb.Dataset(train_df[train_df.y_m < test][features], 
# MAGIC                                    label=train_df[train_df.y_m < test][TARGET])
# MAGIC             valid_data = lgb.Dataset(train_df[train_df.y_m == test][features], 
# MAGIC                                    label=train_df[train_df.y_m == test][TARGET])
# MAGIC             estimator = lgb.train(lgb_params,
# MAGIC                                   train_data,
# MAGIC                                   valid_sets = [valid_data],
# MAGIC                                   num_boost_round=num,
# MAGIC                                   early_stopping_rounds=40,
# MAGIC                                 verbose_eval=-1                             
# MAGIC                                  )
# MAGIC 
# MAGIC             error = wape(estimator.predict(train_df[train_df.y_m == test1][features]), train_df.loc[train_df.y_m == test1, TARGET])
# MAGIC             errors.append(error)    
# MAGIC        
# MAGIC         col_error = np.mean(np.array(errors))
# MAGIC         if col_error < best_col_error:
# MAGIC             best_col = col
# MAGIC             best_col_error = col_error
# MAGIC     print(features)    
# MAGIC     print('Лучшая средняя ошибка M2:')
# MAGIC     print(best_col_error)
# MAGIC     WAPEmean[q] = np.mean(np.array(errors))
# MAGIC 
# MAGIC     used_cols.append(best_col)
# MAGIC     rest_cols = [x for x in rest_cols if x != best_col]
# MAGIC     q += 1        

# COMMAND ----------

used_cols

# COMMAND ----------

# MAGIC %%time
# MAGIC WAPEmean = np.zeros(len(range(25,501,25)))
# MAGIC NumIter = np.zeros(len(range(25,501,25)))
# MAGIC q = 0
# MAGIC for n in range(25,501,25):
# MAGIC     # LightGBM тест ООF на +1
# MAGIC     errors = []
# MAGIC     train_df['OOF'] = 0
# MAGIC     feature_importance_df = pd.DataFrame()
# MAGIC     print(n)
# MAGIC     num=n
# MAGIC     YL['OOF1a'] = 0
# MAGIC     YL['OOF2a'] = 0
# MAGIC     YL['OOF3a'] = 0
# MAGIC     for i in range(1,6):
# MAGIC         print('M1')
# MAGIC         features = used_cols
# MAGIC         test_m = 201912
# MAGIC         test = base.iloc[-i-4,1]
# MAGIC         test1 = base.iloc[-i-3,1]
# MAGIC         print(test)
# MAGIC         train_data = lgb.Dataset(train_df[train_df.y_m < test][features], 
# MAGIC                                label=train_df[train_df.y_m < test][TARGET])
# MAGIC         valid_data = lgb.Dataset(train_df[train_df.y_m == test][features], 
# MAGIC                                label=train_df[train_df.y_m == test][TARGET])
# MAGIC         estimator = lgb.train(lgb_params,
# MAGIC                               train_data,
# MAGIC                               valid_sets = [valid_data],
# MAGIC                               num_boost_round=num,
# MAGIC                               early_stopping_rounds=n/5,
# MAGIC                             verbose_eval=1000                             
# MAGIC                              )
# MAGIC 
# MAGIC         YL['OOF1a'] += estimator.predict(YL[features]) / 5
# MAGIC         error = wape(estimator.predict(train_df[train_df.y_m == test1][features]), train_df.loc[train_df.y_m == test1, TARGET])
# MAGIC         errors.append(error)    
# MAGIC #         print(error)
# MAGIC         importance_df = pd.DataFrame()
# MAGIC         importance_df["feature"] = features
# MAGIC         importance_df["importance"] = estimator.feature_importance()
# MAGIC         feature_importance_df = pd.concat([feature_importance_df, importance_df], axis=0)
# MAGIC 
# MAGIC     print('Средняя ошибка M2:')
# MAGIC     print(np.mean(np.array(errors)))
# MAGIC     WAPEmean[q] = np.mean(np.array(errors))
# MAGIC     NumIter[q] = n
# MAGIC     q += 1
# MAGIC i_min = np.argmin(WAPEmean)
# MAGIC print(i_min)
# MAGIC print(WAPEmean[i_min])
# MAGIC print(NumIter[i_min])

# COMMAND ----------

features2 = used_cols
num2 = NumIter[i_min]

# COMMAND ----------

display_importances(feature_importance_df)

# COMMAND ----------

TARGET  = 'Q' 
features3 = ['code',  'Price', 
#        'SalesQ_lag1', 
#              'SalesM_lag1',
#              'SalesQ_lag2', 'SalesM_lag2',
       'SalesQ_lag3', 'SalesM_lag3', 'SalesQ_lag4', 'SalesM_lag4',
       'SalesQ_lag5', 'SalesM_lag5', 
#              'rolling_mean_Q1_2', 'rolling_std_Q1_2',
#        'rolling_mean_Money1_2', 'rolling_std_Money1_2', 'rolling_mean_Q1_3',
#        'rolling_std_Q1_3', 'rolling_mean_Money1_3', 'rolling_std_Money1_3',
#        'rolling_mean_Q1_6', 'rolling_std_Q1_6', 'rolling_mean_Money1_6',
#        'rolling_std_Money1_6', 
#              'rolling_mean_Q1_9', 'rolling_std_Q1_9',
#        'rolling_mean_Money1_9', 'rolling_std_Money1_9', 
#              'rolling_mean_Q2_2',
#        'rolling_std_Q2_2', 'rolling_mean_Money2_2', 'rolling_std_Money2_2',
#        'rolling_mean_Q2_3', 'rolling_std_Q2_3', 'rolling_mean_Money2_3',
#        'rolling_std_Money2_3', 'rolling_mean_Q2_6', 'rolling_std_Q2_6',
#        'rolling_mean_Money2_6', 'rolling_std_Money2_6', 
#              'rolling_mean_Q2_9',
#        'rolling_std_Q2_9', 'rolling_mean_Money2_9', 'rolling_std_Money2_9',
       'rolling_mean_Q3_2', 'rolling_std_Q3_2', 'rolling_mean_Money3_2',
       'rolling_std_Money3_2', 'rolling_mean_Q3_3', 'rolling_std_Q3_3',
       'rolling_mean_Money3_3', 'rolling_std_Money3_3', 'rolling_mean_Q3_6',
       'rolling_std_Q3_6', 'rolling_mean_Money3_6', 'rolling_std_Money3_6',
#        'rolling_mean_Q3_9', 'rolling_std_Q3_9', 'rolling_mean_Money3_9',
#        'rolling_std_Money3_9', 
             'CountBrand', 'year', 'month', 'y_m',
       'encBrand_mean', 'encBrand_std', 'Price_lag', 'Price_change',
       'Price_grow']

# COMMAND ----------

# MAGIC %%time
# MAGIC rest_cols = features3
# MAGIC WAPEmean = np.zeros(len(rest_cols))
# MAGIC NumIter = np.zeros(len(rest_cols))
# MAGIC q = 0
# MAGIC used_cols = []
# MAGIC while len(rest_cols) > 0:
# MAGIC     best_col = ''
# MAGIC     best_col_error = 1e6
# MAGIC     for col in rest_cols:
# MAGIC         features = used_cols + [col]
# MAGIC         # LightGBM тест ООF на +1
# MAGIC         errors = []
# MAGIC         num=200
# MAGIC         for i in range(1,6):
# MAGIC             test_m = 201912
# MAGIC             test = base.iloc[-i-4,1]
# MAGIC             test1 = base.iloc[-i-3,1]
# MAGIC             print(test)
# MAGIC             train_data = lgb.Dataset(train_df[train_df.y_m < test][features], 
# MAGIC                                    label=train_df[train_df.y_m < test][TARGET])
# MAGIC             valid_data = lgb.Dataset(train_df[train_df.y_m == test][features], 
# MAGIC                                    label=train_df[train_df.y_m == test][TARGET])
# MAGIC             estimator = lgb.train(lgb_params,
# MAGIC                                   train_data,
# MAGIC                                   valid_sets = [valid_data],
# MAGIC                                   num_boost_round=num,
# MAGIC                                   early_stopping_rounds=40,
# MAGIC                                 verbose_eval=-1                             
# MAGIC                                  )
# MAGIC 
# MAGIC             error = wape(estimator.predict(train_df[train_df.y_m == test1][features]), train_df.loc[train_df.y_m == test1, TARGET])
# MAGIC             errors.append(error)    
# MAGIC        
# MAGIC         col_error = np.mean(np.array(errors))
# MAGIC         if col_error < best_col_error:
# MAGIC             best_col = col
# MAGIC             best_col_error = col_error
# MAGIC     print(features)    
# MAGIC     print('Лучшая средняя ошибка M3:')
# MAGIC     print(best_col_error)
# MAGIC     WAPEmean[q] = np.mean(np.array(errors))
# MAGIC 
# MAGIC     used_cols.append(best_col)
# MAGIC     rest_cols = [x for x in rest_cols if x != best_col]
# MAGIC     q += 1

# COMMAND ----------

i_min = np.argmin(WAPEmean)
print(i_min)
print(WAPEmean[i_min])

# COMMAND ----------

# MAGIC %%time
# MAGIC rest_cols = features3
# MAGIC WAPEmean = np.zeros(len(rest_cols))
# MAGIC NumIter = np.zeros(len(rest_cols))
# MAGIC q = 0
# MAGIC used_cols = []
# MAGIC while q < i_min + 1:
# MAGIC     best_col = ''
# MAGIC     best_col_error = 1e6
# MAGIC     for col in rest_cols:
# MAGIC         features = used_cols + [col]
# MAGIC         # LightGBM тест ООF на +1
# MAGIC         errors = []
# MAGIC         num=200
# MAGIC         for i in range(1,6):
# MAGIC             test_m = 201912
# MAGIC             test = base.iloc[-i-4,1]
# MAGIC             test1 = base.iloc[-i-3,1]
# MAGIC             print(test)
# MAGIC             train_data = lgb.Dataset(train_df[train_df.y_m < test][features], 
# MAGIC                                    label=train_df[train_df.y_m < test][TARGET])
# MAGIC             valid_data = lgb.Dataset(train_df[train_df.y_m == test][features], 
# MAGIC                                    label=train_df[train_df.y_m == test][TARGET])
# MAGIC             estimator = lgb.train(lgb_params,
# MAGIC                                   train_data,
# MAGIC                                   valid_sets = [valid_data],
# MAGIC                                   num_boost_round=num,
# MAGIC                                   early_stopping_rounds=40,
# MAGIC                                 verbose_eval=-1                             
# MAGIC                                  )
# MAGIC 
# MAGIC             error = wape(estimator.predict(train_df[train_df.y_m == test1][features]), train_df.loc[train_df.y_m == test1, TARGET])
# MAGIC             errors.append(error)    
# MAGIC        
# MAGIC         col_error = np.mean(np.array(errors))
# MAGIC         if col_error < best_col_error:
# MAGIC             best_col = col
# MAGIC             best_col_error = col_error
# MAGIC     print(features)    
# MAGIC     print('Лучшая средняя ошибка M3:')
# MAGIC     print(best_col_error)
# MAGIC     WAPEmean[q] = np.mean(np.array(errors))
# MAGIC 
# MAGIC     used_cols.append(best_col)
# MAGIC     rest_cols = [x for x in rest_cols if x != best_col]
# MAGIC     q += 1    

# COMMAND ----------

used_cols

# COMMAND ----------

# MAGIC %%time
# MAGIC WAPEmean = np.zeros(len(range(25,501,25)))
# MAGIC NumIter = np.zeros(len(range(25,501,25)))
# MAGIC q = 0
# MAGIC for n in range(25,501,25):
# MAGIC     # LightGBM тест ООF на +1
# MAGIC     errors = []
# MAGIC     train_df['OOF'] = 0
# MAGIC     feature_importance_df = pd.DataFrame()
# MAGIC     print(n)
# MAGIC     num=n
# MAGIC     YL['OOF1a'] = 0
# MAGIC     YL['OOF2a'] = 0
# MAGIC     YL['OOF3a'] = 0
# MAGIC     for i in range(1,6):
# MAGIC         features = used_cols
# MAGIC         test_m = 201912
# MAGIC         test = base.iloc[-i-4,1]
# MAGIC         test1 = base.iloc[-i-3,1]
# MAGIC         print(test)
# MAGIC         train_data = lgb.Dataset(train_df[train_df.y_m < test][features], 
# MAGIC                                label=train_df[train_df.y_m < test][TARGET])
# MAGIC         valid_data = lgb.Dataset(train_df[train_df.y_m == test][features], 
# MAGIC                                label=train_df[train_df.y_m == test][TARGET])
# MAGIC         estimator = lgb.train(lgb_params,
# MAGIC                               train_data,
# MAGIC                               valid_sets = [valid_data],
# MAGIC                               num_boost_round=num,
# MAGIC                               early_stopping_rounds=n/5,
# MAGIC                             verbose_eval=1000                             
# MAGIC                              )
# MAGIC 
# MAGIC         YL['OOF1a'] += estimator.predict(YL[features]) / 5
# MAGIC         error = wape(estimator.predict(train_df[train_df.y_m == test1][features]), train_df.loc[train_df.y_m == test1, TARGET])
# MAGIC         errors.append(error)    
# MAGIC #         print(error)
# MAGIC         importance_df = pd.DataFrame()
# MAGIC         importance_df["feature"] = features
# MAGIC         importance_df["importance"] = estimator.feature_importance()
# MAGIC         feature_importance_df = pd.concat([feature_importance_df, importance_df], axis=0)
# MAGIC 
# MAGIC     print('Средняя ошибка M3:')
# MAGIC     print(np.mean(np.array(errors)))
# MAGIC     WAPEmean[q] = np.mean(np.array(errors))
# MAGIC     NumIter[q] = n
# MAGIC     q += 1
# MAGIC i_min = np.argmin(WAPEmean)
# MAGIC print(i_min)
# MAGIC print(WAPEmean[i_min])
# MAGIC print(NumIter[i_min])

# COMMAND ----------

features3 = used_cols
num3 = NumIter[i_min]

# COMMAND ----------

display_importances(feature_importance_df)

# COMMAND ----------

# MAGIC %%time
# MAGIC 
# MAGIC # финальный расчет по всем LightGBM валидация на +1
# MAGIC errors = []
# MAGIC YL['OOF'] = 0
# MAGIC feature_importance_df = pd.DataFrame()
# MAGIC 
# MAGIC for i in range(1,7):
# MAGIC     print('M1')
# MAGIC     features = features1
# MAGIC     test_m = 202001
# MAGIC     test = base.iloc[-i-3,1]
# MAGIC     test1 = base.iloc[-i-3,1]
# MAGIC     print(test)
# MAGIC     print(num1/5)
# MAGIC     train_data = lgb.Dataset(train_df[train_df.y_m < test][features], 
# MAGIC                            label=train_df[train_df.y_m < test][TARGET])
# MAGIC     valid_data = lgb.Dataset(train_df[train_df.y_m == test][features], 
# MAGIC                            label=train_df[train_df.y_m == test][TARGET])
# MAGIC     estimator = lgb.train(lgb_params,
# MAGIC                           train_data,
# MAGIC                           valid_sets = [valid_data],
# MAGIC                           num_boost_round=int(num1),
# MAGIC                           early_stopping_rounds=int(num1/5),
# MAGIC                         verbose_eval=100                             
# MAGIC                          )
# MAGIC     
# MAGIC     YL.loc[YL.y_m == test_m,'OOF'] += estimator.predict(YL[YL.y_m == test_m][features]) / 6
# MAGIC     error = wape(estimator.predict(train_df[train_df.y_m == test1][features]), train_df.loc[train_df.y_m == test1, TARGET])
# MAGIC     errors.append(error)    
# MAGIC     print(error)
# MAGIC     importance_df = pd.DataFrame()
# MAGIC     importance_df["feature"] = features
# MAGIC     importance_df["importance"] = estimator.feature_importance()
# MAGIC     feature_importance_df = pd.concat([feature_importance_df, importance_df], axis=0)
# MAGIC     
# MAGIC print('Средняя ошибка M1:')
# MAGIC print(np.mean(np.array(errors)))
# MAGIC 
# MAGIC 
# MAGIC for i in range(1,7):
# MAGIC     print('M2')
# MAGIC     features = features2
# MAGIC     test_m = 202002
# MAGIC     test = base.iloc[-i-3,1]
# MAGIC     test1 = base.iloc[-i-3,1]
# MAGIC     print(test)
# MAGIC     train_data = lgb.Dataset(train_df[train_df.y_m < test][features], 
# MAGIC                            label=train_df[train_df.y_m < test][TARGET])
# MAGIC     valid_data = lgb.Dataset(train_df[train_df.y_m == test][features], 
# MAGIC                            label=train_df[train_df.y_m == test][TARGET])
# MAGIC     estimator = lgb.train(lgb_params,
# MAGIC                           train_data,
# MAGIC                           valid_sets = [valid_data],
# MAGIC                           num_boost_round=int(num2),
# MAGIC                           early_stopping_rounds=int(num2/5),
# MAGIC                         verbose_eval=100                             
# MAGIC                          )
# MAGIC     
# MAGIC     YL.loc[YL.y_m == test_m,'OOF'] += estimator.predict(YL[YL.y_m == test_m][features]) / 6
# MAGIC     error = wape(estimator.predict(train_df[train_df.y_m == test1][features]), train_df.loc[train_df.y_m == test1, TARGET])
# MAGIC     errors.append(error)
# MAGIC     print(error)
# MAGIC 
# MAGIC for i in range(1,7):
# MAGIC     print('M3')
# MAGIC     features = features3
# MAGIC     test_m = 202003
# MAGIC     test = base.iloc[-i-3,1]
# MAGIC     test1 = base.iloc[-i-3,1]
# MAGIC     print(test)
# MAGIC     train_data = lgb.Dataset(train_df[train_df.y_m < test][features], 
# MAGIC                            label=train_df[train_df.y_m < test][TARGET])
# MAGIC     valid_data = lgb.Dataset(train_df[train_df.y_m == test][features], 
# MAGIC                            label=train_df[train_df.y_m == test][TARGET])
# MAGIC     estimator = lgb.train(lgb_params,
# MAGIC                           train_data,
# MAGIC                           valid_sets = [valid_data],
# MAGIC                           num_boost_round=int(num3),
# MAGIC                           early_stopping_rounds=int(num3/5),
# MAGIC                         verbose_eval=100                             
# MAGIC                          )
# MAGIC     
# MAGIC     YL.loc[YL.y_m == test_m,'OOF'] += estimator.predict(YL[YL.y_m == test_m][features]) / 6
# MAGIC     error = wape(estimator.predict(train_df[train_df.y_m == test1][features]), train_df.loc[train_df.y_m == test1, TARGET])
# MAGIC     errors.append(error)
# MAGIC     print(error)
# MAGIC     
# MAGIC print('Средняя ошибка по всем:')
# MAGIC print(np.mean(np.array(errors)))

# COMMAND ----------

display_importances(feature_importance_df)

# COMMAND ----------

test_res = YL[YL.y_m > 201912][['code', 'date', 'Brand', 'Product','y_m','Date','OOF']]
test_res.to_csv('../Results/20200922RezultGSK_M31.csv', index=False)

# COMMAND ----------

