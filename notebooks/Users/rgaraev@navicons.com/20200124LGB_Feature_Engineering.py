# Databricks notebook source
lgb_params_tune = {
  'boosting_type':['gbdt'],
  'objective': ['regression'],
  'metric': ['l2'],
  'num_leaves': [8, 32,64],
  'max_depth': [3,4,6],
  'learning_rate': [0.01, 0.05, 0.1],
  'feature_fraction': [0.3, 0.6, 0.9],
  'bagging_fraction': [0.2, 0.4, 0.8],
  'bagging_freq': [3, 5, 7],
  'verbose':[0]
}

# COMMAND ----------

# General imports
import numpy as np
import pandas as pd
import os, sys, gc, time, warnings, pickle, psutil, random

# custom imports
import decimal
from multiprocessing import Pool        # Multiprocess Runs
from functools import partial
from sklearn.ensemble import VotingRegressor,RandomForestRegressor,AdaBoostRegressor,StackingRegressor
from sklearn.neighbors import KNeighborsRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime as dt
from plotly.offline import init_notebook_mode, iplot
from sklearn.model_selection import TimeSeriesSplit # you have everything done for you
from plotly import graph_objs as go
import statsmodels.api as sm
# warnings.filterwarnings('ignore')
import lightgbm as lgb
from scipy import stats
#from fbprophet import Prophet
import itertools
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
#from fbprophet.diagnostics import cross_validation

# COMMAND ----------

# MAGIC %sh
# MAGIC wget -O azcopy.tar.gz https://aka.ms/downloadazcopy-v10-linux
# MAGIC tar -xf azcopy.tar.gz
# MAGIC cp "$(dirname "$(find . -path ./azcopy_linux\* -type f| tail -1)")"/azcopy azcopy
# MAGIC mkdir data
# MAGIC mkdir model
# MAGIC ./azcopy copy "https://bricksresult.blob.core.windows.net/data/SalesLetoil/*?sv=2019-12-12&ss=bfqt&srt=sco&sp=rwdlacupx&se=2028-01-11T21:35:49Z&st=2021-01-12T13:35:49Z&spr=https&sig=%2F9iDTc%2FE8MOMfEAvfTYkTpil%2BR5fLzdLkBJyDbJsfWw%3D" "data/"  

# COMMAND ----------

# функция WAPE
def wape(y_pred, y_true):
    res = np.sum(np.abs(y_true - y_pred)) / np.abs(np.sum(y_true)) * 100
    return res
  
def read_parquet_folder_as_pandas(path, verbosity=1):
  files = [f for f in os.listdir(path) if f.endswith("parquet")]

  if verbosity > 0:
    print("{} parquet files found. Beginning reading...".format(len(files)), end="")
    start = datetime.datetime.now()

  df_list = [pd.read_parquet(os.path.join(path, f)) for f in files]
  df = pd.concat(df_list, ignore_index=True)

  if verbosity > 0:
    end = datetime.datetime.now()
    print(" Finished. Took {}".format(end-start))
  return df

# COMMAND ----------

df = read_parquet_folder_as_pandas('data')
df.drop(df[df.salesQ==0].index,inplace=True)
df['timestamp'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str),
                                              format='%Y-%m')
df.head()

# COMMAND ----------

df['price'] =  df['salesM'].astype(int) / df['salesQ'].astype(int)
df['price_lag'] = df['price'].shift(1)
df.drop(df[df.price_lag.isnull()].index,inplace=True)

for i in range(1,6):
  df['SalesQ_lag'+str(i)] = df.groupby(['ITEMID'])['salesQ'].transform(lambda x: x.shift(i))
  df['SalesM_lag'+str(i)] = df.groupby(['ITEMID'])['salesM'].transform(lambda x: x.shift(i))
 
for i in range(1,4):
  for j in [2,3,6,9]:
    df['rolling_mean_Q'+str(i)+'_'+str(j)] = df.groupby(['ITEMID'])['salesQ'].transform(lambda x: x.shift(i).rolling(j).mean())
    df['rolling_std_Q'+str(i)+'_'+str(j)] = df.groupby(['ITEMID'])['salesQ'].transform(lambda x: x.shift(i).rolling(j).std())
    df['rolling_mean_M'+str(i)+'_'+str(j)] = df.groupby(['ITEMID'])['salesM'].transform(lambda x: x.shift(i).rolling(j).mean())
    df['rolling_std_M'+str(i)+'_'+str(j)] = df.groupby(['ITEMID'])['salesM'].transform(lambda x: x.shift(i).rolling(j).std())
    
 
 
lgb_params = {
  'boosting_type': 'gbdt',
  'objective': 'regression',
  'metric': 'l2',
  'num_leaves': 31,
  'learning_rate': 0.1,
  'feature_fraction': 0.9,
  'bagging_fraction': 0.8,
  'bagging_freq': 5,
  'verbose': 0
}
 
def display_importances(feature_importance_df_):
  cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:42].index
  best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
  plt.figure(figsize=(8, 10))
  sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
  plt.title('LightGBM Features (averaged predictions)')
  plt.tight_layout()
  
TARGET = 'salesQ'
 
df['y_m'] = df.year.astype(str)+df.month.astype(str)
graf = df.groupby(['y_m'])['year'].first().reset_index()
base = graf['y_m'].reset_index(drop=True).reset_index()
base.columns = ['wm_yr_wk','y_m']
 
base.drop( (base[ (base['y_m'].astype(int) == 201912)].index  ),inplace=True   )
base.drop( (base[ (base['y_m'].astype(int) == 201911)].index  ),inplace=True   )
base.drop( (base[ (base['y_m'].astype(int) == 201910)].index  ),inplace=True   )
base.drop( (base[ (base['y_m'].astype(int) == 20199)].index  ),inplace=True   )
 
df['ITEMID'] = df['ITEMID'].astype('category')
df['y_m'] = df['y_m'].astype('str')
object_cols = df.select_dtypes('object').columns
df[object_cols] = df[object_cols].astype(float)

# COMMAND ----------

useful_features = ['SalesQ_lag1', 'SalesM_lag4', 'rolling_mean_M1_3', 'rolling_std_M2_9', 'price', 'rolling_mean_M2_3', 'rolling_std_M1_9', 'rolling_mean_M1_2', 'SalesM_lag2', 'rolling_std_M1_6', 'rolling_std_M3_6', 'rolling_mean_M2_9', 'rolling_std_M2_2', 'rolling_std_M1_2', 'rolling_std_M3_2', 'rolling_mean_M1_6', 'rolling_std_M3_3', 'rolling_mean_M3_6', 'rolling_mean_M2_6']

TARGET='salesQ'

# COMMAND ----------

df['month*price'] = df['month'] * df['price']
df['month*price_lag'] = df['month'] * df['price_lag']
df['price_lag*price'] = df['price_lag'] * df['price']
df['month*month'] = df['month'] * df['month']
df['price*price'] = df['price'] * df['price']
df['price_lag*price_lag'] = df['price_lag'] * df['price_lag']
df['wape_indicator'] = np.where(wape(df.price_lag , df.price)>75,1,0)

helper = df.copy()
helper.set_index('timestamp',inplace=True)

sum_15_q = helper[:'2015-12-12'].salesQ.sum()
sum_15_m = helper[:'2015-12-12'].salesM.sum()

sum_16_q = helper['2015-12-12':'2016-12-12'].salesQ.sum()
sum_16_m = helper['2015-12-12':'2016-12-12'].salesQ.sum()

sum_17_q = helper['2016-12-12':'2017-12-12'].salesQ.sum()
sum_17_m = helper['2016-12-12':'2017-12-12'].salesQ.sum()

sum_18_q = helper['2017-12-12':'2018-12-12'].salesQ.sum()
sum_18_m = helper['2017-12-12':'2018-12-12'].salesQ.sum()

sum_19_q = helper['2018-12-12':'2019-12-12'].salesQ.sum()
sum_19_m = helper['2018-12-12':'2019-12-12'].salesQ.sum()

warehouseQ = np.zeros((df.shape[0]))
warehouseM = np.zeros((df.shape[0]))

for index,month,year in zip(range(0,df.shape[0]),df.month,df.year):
  
  if (year==2015):
    warehouseQ[index] = sum_15_q / df[ (df['month']==month) & (df['year']==2015)].salesQ.sum()   
    warehouseQ[index] = sum_15_m / df[ (df['month']==month) & (df['year']==2015)].salesM.sum()   
    
  elif (year==2016):
    warehouseQ[index] = sum_16_q / df[ (df['month']==month) & (df['year']==2016)].salesQ.sum()   
    warehouseQ[index] = sum_16_m / df[ (df['month']==month) & (df['year']==2016)].salesM.sum()   
  
  elif (year==2017):
    warehouseQ[index] = sum_17_q / df[ (df['month']==month) & (df['year']==2017)].salesQ.sum()   
    warehouseQ[index] = sum_17_m / df[ (df['month']==month) & (df['year']==2017)].salesM.sum()   
    
  elif (year==2018):
    warehouseQ[index] = sum_18_q / df[ (df['month']==month) & (df['year']==2018)].salesQ.sum()   
    warehouseQ[index] = sum_18_m / df[ (df['month']==month) & (df['year']==2018)].salesM.sum()   
    
  elif (year==2019):
    warehouseQ[index] = sum_19_q / df[ (df['month']==month) & (df['year']==2019)].salesQ.sum()   
    warehouseQ[index] = sum_19_m / df[ (df['month']==month) & (df['year']==2019)].salesM.sum()   
    
df['y_m_salesQ'] = warehouseQ
df['y_m_salesM'] = warehouseM

train_df = df[df['timestamp'] < '2019-09-01']
test_df = df[df['timestamp'] >= '2019-09-01']

extra_features = ['month*price', 'month*price_lag',
       'price_lag*price', 'month*month', 'price*price', 'price_lag*price_lag','wape_indicator','y_m_salesQ','y_m_salesM']

# COMMAND ----------

rest_cols = extra_features
WAPEmean = np.zeros(len(rest_cols))
NumIter = np.zeros(len(rest_cols))
q = 0
used_cols = []
while len(rest_cols) > 0 :
    best_col = ''
    best_col_error = 1e6
    for col in rest_cols:
        features = used_cols + [col]
        # LightGBM тест ООF на +1
        errors = []
        num=200
        for i in range(1,6):
            test_m = 201912
            test = base.iloc[-i-4,1]
            test1 = base.iloc[-i-3,1]
            print(test)
            train_data = lgb.Dataset(train_df[train_df.y_m.astype('int') < int(test)][features], 
                                   label=train_df[train_df.y_m.astype('int') < int(test)][TARGET])
            valid_data = lgb.Dataset(train_df[train_df.y_m.astype('int') == int(test)][features], 
                                   label=train_df[train_df.y_m.astype('int') == int(test)][TARGET])
            estimator = lgb.train(lgb_params,
                                  train_data,
                                  valid_sets = [valid_data],
                                 num_boost_round=num,
                                  early_stopping_rounds=40,
                                verbose_eval=-1                             
                                 )

            error = wape(estimator.predict(train_df[train_df.y_m.astype('int') == int(test1)][features]), train_df.loc[train_df.y_m.astype('int') == int(test1), TARGET])
            errors.append(error)    
       
        col_error = np.mean(np.array(errors))
        if col_error < best_col_error:
            best_col = col
            best_col_error = col_error
    print(features)    
    print('Лучшая средняя ошибка M1:')
    print(best_col_error)
    WAPEmean[q] = np.mean(np.array(errors))

    used_cols.append(best_col)
    
    rest_cols = [x for x in rest_cols if x != best_col]
    q += 1        

    

# COMMAND ----------

print(extra_features)

i_min = np.argmin(WAPEmean)
print(i_min)
print(WAPEmean[i_min])
print(WAPEmean)

# ['month*price', 'month*price_lag', 'price_lag*price', 'month*month', 'price*price', 'price_lag*price_lag', 'wape_indicator', 'y_m_salesQ', 'y_m_salesM']
# 1
# 136.45895179614192
# [143.46528556 136.4589518  136.4589518  153.52249086 146.25456556
#  148.31067653 172.3191951  149.74131357 168.75875837]

# COMMAND ----------

features = ['SalesQ_lag1', 'SalesM_lag4', 'rolling_mean_M1_3', 'rolling_std_M2_9', 'price', 'rolling_mean_M2_3', 'SalesM_lag2',  'rolling_std_M3_6', 'rolling_mean_M2_9', 'rolling_std_M2_2', 'rolling_std_M1_2', 'rolling_std_M3_2', 'rolling_mean_M1_6', 'rolling_std_M3_3', 'rolling_mean_M3_6', 'rolling_mean_M2_6','month*price_lag']

# COMMAND ----------

train_data = lgb.Dataset(train_df[train_df.y_m.astype('int') < int(20199)][features], 
                                   label=train_df[train_df.y_m.astype('int') < int(20199)][TARGET])
test_data = lgb.Dataset(train_df[train_df.y_m.astype('int') >= int(20199)][features], 
                                   label=train_df[train_df.y_m.astype('int') >= int(20199)][TARGET])

model = lgb.train(lgb_params,train_data,  num_boost_round=250)

preds = model.predict(train_df[train_df.y_m.astype('int') >= int(20199)][features])

wape(preds, train_df[train_df.y_m.astype('int') >= int(20199)][TARGET]) # 44

# COMMAND ----------

nums = list(range(20,400,20))
res = []
for i in nums:
          train_data = lgb.Dataset(train_df[train_df.y_m.astype('int') < int(20199)][features], 
                                               label=train_df[train_df.y_m.astype('int') < int(20199)][TARGET])
          test_data = lgb.Dataset(train_df[train_df.y_m.astype('int') >= int(20199)][features], 
                                               label=train_df[train_df.y_m.astype('int') >= int(20199)][TARGET])
          model = lgb.train(lgb_params,train_data,  num_boost_round=i)
          preds = model.predict(train_df[train_df.y_m.astype('int') >= int(20199)][features])
          metric = wape(preds, train_df[train_df.y_m.astype('int') >= int(20199)][TARGET])
          res.append(metric)
          
plt.figure(figsize=(10,10))
plt.title('wape through rounds')
plt.plot(nums,res)

# COMMAND ----------

