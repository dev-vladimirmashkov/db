# Databricks notebook source
import numpy as np
import pandas as pd
import os, sys, gc, time, warnings, pickle, psutil, random
# custom imports
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
warnings.filterwarnings('ignore')
import lightgbm as lgb
from scipy import stats
import itertools
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
import calendar
from prophet import Prophet
import logging
logging.getLogger("py4j").setLevel(logging.ERROR)
from statsmodels.tsa.holtwinters import ExponentialSmoothing
#import seasonal
pd.__version__

# COMMAND ----------

pip install prophet

# COMMAND ----------

# MAGIC %sh
# MAGIC wget -O azcopy.tar.gz https://aka.ms/downloadazcopy-v10-linux
# MAGIC tar -xf azcopy.tar.gz
# MAGIC cp "$(dirname "$(find . -path ./azcopy_linux\* -type f| tail -1)")"/azcopy azcopy
# MAGIC mkdir data
# MAGIC mkdir model
# MAGIC ./azcopy copy "https://bricksresult.blob.core.windows.net/data/SalesLetoil/*?sv=2019-12-12&ss=bfqt&srt=sco&sp=rwdlacupx&se=2028-01-11T21:35:49Z&st=2021-01-12T13:35:49Z&spr=https&sig=%2F9iDTc%2FE8MOMfEAvfTYkTpil%2BR5fLzdLkBJyDbJsfWw%3D" "data/"  

# COMMAND ----------

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

df = read_parquet_folder_as_pandas('data')
df = df.dropna()
df['month'] = df['month'].astype(int)
df['year'] = df['year'].astype(int)
df['timestamp'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str),
                                              format='%Y-%m')
df = df.sort_values(by='timestamp')
df.drop(  df[df.salesQ==0].index, inplace=True)
df['salesM'] = df['salesM'].astype(float)
df['salesQ'] = df['salesQ'].astype(float)
df = df.drop(df[df.salesQ<0].index)
df.iloc[:,[4,5]].groupby(['timestamp']).sum().plot(title = 'Timestamp and SalesQ')
df = df[df.timestamp>='2017-01-01']

# COMMAND ----------

def determine_gaps(df, name):
  curr_df = df[df.ITEMID==name].reset_index(drop=True).sort_values('timestamp')
  first = curr_df.loc[0,'timestamp']
  last = curr_df.loc[curr_df.shape[0]-1,'timestamp']
  ranges_shape = pd.date_range(first, last, freq = 'MS').shape[0]
  if ranges_shape != curr_df.shape[0]:
    return False
  
def fill_gaps(df,name):
  curr_df = df[df.ITEMID==name].reset_index(drop=True).sort_values('timestamp')
  first = curr_df.loc[0,'timestamp']
  last = curr_df.loc[curr_df.shape[0]-1,'timestamp']
  true_vals = set(pd.date_range(first, last, freq = 'MS'))
  actual_vals = set(pd.to_datetime(curr_df.timestamp.values))
  residuals = list(true_vals - actual_vals)
  add_ad = pd.DataFrame(columns = ['ITEMID','salesQ','timestamp'] )
  add_ad['timestamp'] = residuals
  add_ad['ITEMID'] = name
  add_ad['salesQ'] = np.nan
  curr_df = curr_df.append(add_ad)
  curr_df = curr_df.sort_values('timestamp').reset_index(drop=True)
  curr_df = curr_df.fillna(method='backfill')
  return curr_df

def splitf(df):
  size = df.shape[0]
  if (size <=12):
      train = df.iloc[:-3,:]
      test = df.iloc[-3:,:]
      return train,test
  elif (size >12 and size <=24):
      train = df.iloc[:-6,:]
      test = df.iloc[-6:,:]
      return train,test
  else:
      train = df.iloc[:-9,:]
      test = df.iloc[-9:,:]
      return train,test
    
def wape(y_pred, y_true):
    res = np.sum(np.abs(y_true - y_pred)) / np.abs(np.sum(y_true)) * 100
    return res
  
def make_int_list(l, dtype=int):
    return list(map(dtype, l))

# COMMAND ----------

res_df = df[['ITEMID','salesQ','timestamp']]
res_df = res_df.groupby('ITEMID').filter(lambda x: len(x)>5)
res_df = res_df[res_df.timestamp >= '2017-01-01']
result_s = pd.DataFrame(columns = ['ITEMID','forecast_es','wape_es'])
for counter, i in enumerate(res_df.ITEMID.value_counts().index):
    result_s.loc[counter,'ITEMID'] = i
    curr_duration = res_df[res_df.ITEMID==i].shape[0]
    curr_df = res_df[res_df.ITEMID==i].drop('ITEMID',1).set_index('timestamp')
    train, test = splitf(curr_df)
    explore_df = res_df[res_df.ITEMID==i]
    if determine_gaps(explore_df,i) != False:
      s_periods = train.shape[0]
      if (s_periods >= 12):
        s_periods = 12
      model = ExponentialSmoothing(train, seasonal = 'mul', seasonal_periods=s_periods).fit()
      for_wape = model.predict(start = test.index[0], end = test.index[-1])
      pred = model.predict(start = '2021-06-01 00:00:00',end =  '2022-07-01 00:00:00')
      result_s.loc[counter, 'forecast_es'] = pred.values
      result_s.loc[counter,'wape_es'] = wape(test.values,for_wape.values)
    else:
      explore_df = fill_gaps(explore_df,i)
      train, test = splitf(explore_df)
      train = train.set_index('timestamp').drop('ITEMID',1)
      test = test.set_index('timestamp').drop('ITEMID',1)
      curr_df = explore_df.drop('ITEMID',1).set_index('timestamp')
      s_periods = train.shape[0]
      if (s_periods >= 12):
        s_periods = 12
      model = ExponentialSmoothing(train, seasonal = 'mul', seasonal_periods=s_periods).fit()
      for_wape = model.predict(start = test.index[0], end = test.index[-1])
      pred = model.predict(start = '2021-06-01 00:00:00',end =  '2022-07-01 00:00:00')
      result_s.loc[counter, 'forecast_es'] = pred.values
      result_s.loc[counter,'wape_es'] = wape(test.values,for_wape.values)      

# COMMAND ----------

residuals = df[['ITEMID','salesQ','timestamp']]
residuals = df.groupby('ITEMID').filter(lambda x: len(x)<=5)
residuals_res = pd.DataFrame(columns=['ITEMID','forecast'])

for counter, i in enumerate(residuals.ITEMID.value_counts().index):
  residuals_res.loc[counter,'ITEMID'] = i
  residuals_res.loc[counter,'forecast'] = 13*[residuals[residuals.ITEMID==i].salesQ.mean()]
  
residuals_res = residuals_res.reset_index(drop=True)

# COMMAND ----------

prophet = df[['ITEMID','salesQ','timestamp']]
prophet = prophet.groupby('ITEMID').filter(lambda x: len(x)>5)
very_future = pd.DataFrame((
'2021-06-01 00:00:00',
'2021-07-01 00:00:00',
'2021-08-01 00:00:00',
'2021-09-01 00:00:00',
'2021-10-01 00:00:00',
'2021-11-01 00:00:00',
'2021-12-01 00:00:00',
'2022-01-01 00:00:00',
'2022-02-01 00:00:00',
'2022-03-01 00:00:00',
'2022-04-01 00:00:00',
'2022-05-01 00:00:00',
'2022-06-01 00:00:00'),columns = ['ds'])
very_future.ds = pd.to_datetime(very_future.ds)
very_future['floor'] = 0
very_future['cap'] = 1
result = pd.DataFrame(columns = ['ITEMID','wape_prophet', 'forecast_prophet'])
new_year = pd.DataFrame({
  'holiday': 'new_year',
  'ds': pd.to_datetime(['2016-12-31','2017-12-31', '2018-12-31', '2017-12-31', '2019-12-31', '2020-12-31']),
  'lower_window': -3,
  'upper_window': 2
})
w_day =pd.DataFrame({
  'holiday': 'w_day' ,
  'ds': pd.to_datetime(['2016-04-08','2017-04-08', '2018-04-08', '2017-04-08', '2019-04-08', '2020-04-08']),
  'lower_window': -3,
  'upper_window': 0
})
holidays = pd.concat((w_day,new_year))

for counter, i in tqdm(enumerate(prophet.ITEMID.value_counts().index)):
    curr_duration = prophet.ITEMID.value_counts().values[counter]
    curr_df = prophet[prophet.ITEMID==i].reset_index(drop=True)
    curr_df = curr_df.drop('ITEMID',1)
    curr_df.columns = ['y','ds']
    result.loc[counter,'ITEMID'] = i
    
    if ( curr_duration <=12):
      train_curr_df = curr_df.iloc[:-3,:]
      test_curr_df = curr_df.iloc[-3:,:]
      train_curr_df['cap'] = train_curr_df.y.max() * 5
      test_curr_df['floor'] = test_curr_df.y.min() + 1
      model = Prophet( changepoint_prior_scale= 1,
                      seasonality_prior_scale = 0.01,
                      holidays_prior_scale= 0.01,
                      seasonality_mode='multiplicative',
                      holidays=holidays ,
                      weekly_seasonality=False,
                   daily_seasonality=False,growth='logistic'  ).fit(train_curr_df)
      future = pd.DataFrame(test_curr_df.ds)
      future.columns = ['ds']
      future['floor'] = test_curr_df.y.min() +1
      future['cap'] = test_curr_df.y.max() * 5
      y_pred = model.predict(future)
      
    elif (curr_duration >12 and curr_duration <=24):
      train_curr_df = curr_df.iloc[:-6,:]
      test_curr_df = curr_df.iloc[-6:,:]
      train_curr_df['cap'] = train_curr_df.y.max() * 5
      test_curr_df['floor'] = test_curr_df.y.min() + 1
      model = Prophet( changepoint_prior_scale= 1,
                      seasonality_prior_scale = 0.01,
                      holidays_prior_scale= 0.01,
                      seasonality_mode='multiplicative',
                      holidays=holidays ,
                      weekly_seasonality=False,
                   daily_seasonality=False,growth='logistic'  ).fit(train_curr_df)
      future = pd.DataFrame(test_curr_df.ds)
      future.columns = ['ds']
      future['floor'] = test_curr_df.y.min() +1
      future['cap'] = test_curr_df.y.max() * 5
      y_pred = model.predict(future)
      
    else:
      train_curr_df = curr_df.iloc[:-9,:]
      test_curr_df = curr_df.iloc[-9:,:]
      train_curr_df['cap'] = train_curr_df.y.max() * 5
      test_curr_df['floor'] = test_curr_df.y.min()
      model = Prophet( changepoint_prior_scale= 1,
                      seasonality_prior_scale = 0.01,
                      holidays_prior_scale= 0.01,
                      seasonality_mode='multiplicative',
                      holidays=holidays ,
                      weekly_seasonality=False,
                   daily_seasonality=False ,growth='logistic' ).fit(train_curr_df)
      future = pd.DataFrame(test_curr_df.ds)
      future.columns = ['ds']
      future['floor'] = test_curr_df.y.min() + 1
      future['cap'] = test_curr_df.y.max() * 5
      y_pred = model.predict(future)

    metric = wape(y_pred.yhat,test_curr_df.y.values)
    result.loc[counter,'wape_prophet'] = metric
    very_future['floor'] = test_curr_df.y.min() + 1
    very_future['cap'] = test_curr_df.y.max() * 10
    final_preds = model.predict(very_future).yhat.values
    result.loc[counter, 'forecast_prophet'] = final_preds     

# COMMAND ----------

forecasting = result_s.set_index('ITEMID').join(result.set_index('ITEMID')).reset_index()
forecasting['forecast'] = forecasting['forecast_es']

for index, content in forecasting.iterrows():
  if forecasting.loc[index,'wape_prophet'] < forecasting.loc[index,'wape_es']:
    forecasting.loc[index,'forecast'] = forecasting.loc[index,'forecast_prophet']
  else:
    forecasting.loc[index,'forecast'] = forecasting.loc[index,'forecast_es']
forecasting = forecasting.drop(['wape_prophet','wape_es','forecast_es','forecast_prophet'],1)

for index, content in forecasting.iterrows():
  forecasting['forecast'] = forecasting['forecast'].apply(lambda x : np.rint(x))
  
forecasting = pd.concat((forecasting,residuals_res))

# COMMAND ----------

forecasting.to_parquet('data/forecast.parquet', index=False)

# COMMAND ----------

# MAGIC %sh
# MAGIC 
# MAGIC ./azcopy copy data/forecast.parquet "https://bricksresult.blob.core.windows.net/data/forecast.parquet?sv=2019-12-12&ss=bfqt&srt=sco&sp=rwdlacupx&se=2028-01-11T21:35:49Z&st=2021-01-12T13:35:49Z&spr=https&sig=%2F9iDTc%2FE8MOMfEAvfTYkTpil%2BR5fLzdLkBJyDbJsfWw%3D"

# COMMAND ----------

final = pd.read_parquet('data/forecast.parquet')

# COMMAND ----------

# функция WAPE
def wape(y_pred, y_true):
    res = np.sum(np.abs(y_true - y_pred)) / np.abs(np.sum(y_true)) * 100
    return res

for i in range(12,18):
  df['SalesQ_lag'+str(i)] = df.groupby(['ITEMID'])['salesQ'].transform(lambda x: x.shift(i))
  df['SalesM_lag'+str(i)] = df.groupby(['ITEMID'])['salesM'].transform(lambda x: x.shift(i))

for i in range(12,18):
  for j in [2,3,6,9]:
    df['rolling_mean_Q'+str(i)+'_'+str(j)] = df.groupby(['ITEMID'])['salesQ'].transform(lambda x: x.shift(i).rolling(j).mean())
    df['rolling_std_Q'+str(i)+'_'+str(j)] = df.groupby(['ITEMID'])['salesQ'].transform(lambda x: x.shift(i).rolling(j).std())
    df['rolling_mean_M'+str(i)+'_'+str(j)] = df.groupby(['ITEMID'])['salesM'].transform(lambda x: x.shift(i).rolling(j).mean())
    df['rolling_std_M'+str(i)+'_'+str(j)] = df.groupby(['ITEMID'])['salesM'].transform(lambda x: x.shift(i).rolling(j).std())

# COMMAND ----------


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

def display_importances(feature_importance_df_):
  cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:42].index
  best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
  plt.figure(figsize=(8, 10))
  sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
  plt.title('LightGBM Features (averaged predictions)')
  plt.tight_layout()
  
TARGET = 'salesQ'

features_ = ['ITEMID', 'year', 'month', 'price','price_lag','price_change',
       'SalesQ_lag1', 'SalesM_lag1', 'SalesQ_lag2', 'SalesM_lag2',
       'SalesQ_lag3', 'SalesM_lag3', 'SalesQ_lag4', 'SalesM_lag4',
       'SalesQ_lag5', 'SalesM_lag5', 'rolling_mean_Q1_2', 'rolling_std_Q1_2',
       'rolling_mean_M1_2', 'rolling_std_M1_2', 'rolling_mean_Q1_3',
       'rolling_std_Q1_3', 'rolling_mean_M1_3', 'rolling_std_M1_3',
       'rolling_mean_Q1_6', 'rolling_std_Q1_6', 'rolling_mean_M1_6',
       'rolling_std_M1_6', 'rolling_mean_Q1_9', 'rolling_std_Q1_9',
       'rolling_mean_M1_9', 'rolling_std_M1_9', 'rolling_mean_Q2_2',
       'rolling_std_Q2_2', 'rolling_mean_M2_2', 'rolling_std_M2_2',
       'rolling_mean_Q2_3', 'rolling_std_Q2_3', 'rolling_mean_M2_3',
       'rolling_std_M2_3', 'rolling_mean_Q2_6', 'rolling_std_Q2_6',
       'rolling_mean_M2_6', 'rolling_std_M2_6', 'rolling_mean_Q2_9',
       'rolling_std_Q2_9', 'rolling_mean_M2_9', 'rolling_std_M2_9',
       'rolling_mean_Q3_2', 'rolling_std_Q3_2', 'rolling_mean_M3_2',
       'rolling_std_M3_2', 'rolling_mean_Q3_3', 'rolling_std_Q3_3',
       'rolling_mean_M3_3', 'rolling_std_M3_3', 'rolling_mean_Q3_6',
       'rolling_std_Q3_6', 'rolling_mean_M3_6', 'rolling_std_M3_6',
       'rolling_mean_Q3_9', 'rolling_std_Q3_9', 'rolling_mean_M3_9',
       'rolling_std_M3_9']

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

train_df = df[df['timestamp'] < '2019-09-01']
test_df = df[df['timestamp'] >= '2019-09-01']

# COMMAND ----------

rest_cols = features_
WAPEmean = np.zeros(len(rest_cols))
NumIter = np.zeros(len(rest_cols))
q = 0
used_cols = []
while len(used_cols) <= 20:
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

print('useful cols:', used_cols)

i_min = np.argmin(WAPEmean)
print(i_min)
print(WAPEmean[i_min])
print(WAPEmean)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

