# Databricks notebook source
#устанавливаем профет, нужно именно первой строкой тк если после - кластер крашится
!pip install prophet

# COMMAND ----------

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
from sklearn.model_selection import TimeSeriesSplit 
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
from datetime import datetime
from dateutil.relativedelta import relativedelta
logging.getLogger("py4j").setLevel(logging.ERROR)


# COMMAND ----------

# MAGIC %sh
# MAGIC wget -O azcopy.tar.gz https://aka.ms/downloadazcopy-v10-linux
# MAGIC tar -xf azcopy.tar.gz
# MAGIC cp "$(dirname "$(find . -path ./azcopy_linux\* -type f| tail -1)")"/azcopy azcopy
# MAGIC mkdir data
# MAGIC mkdir model
# MAGIC ./azcopy copy "https://alkordatalake.blob.core.windows.net/letoile/raw/ml/SalesLetoil/*?sv=2020-02-10&st=2020-12-31T21%3A00%3A00Z&se=2050-03-31T20%3A59%3A00Z&sr=c&sp=racwdlme&sig=69jYjI73deMsJohT8L9aDGjPE3k9ohHjdzmPPPkTD1s%3D" "data/"  

# COMMAND ----------

# читаем исходные данные с дата лейка
def read_parquet_folder_as_pandas(path, verbosity=1):
  files = [f for f in os.listdir(path) if f.endswith("parquet")]
  df_list = [pd.read_parquet(os.path.join(path, f)) for f in files]
  df = pd.concat(df_list, ignore_index=True)
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
df = df[df.timestamp>='2017-01-01']
# df = df.iloc[:1500,:]

# COMMAND ----------

# вспомагательные функции, метрика - WAPE
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

# хольт винтерс
res_df = df[['ITEMID','salesQ','timestamp']]
res_df = res_df.groupby('ITEMID').filter(lambda x: len(x)>5)
res_df = res_df[res_df.timestamp >= '2017-01-01']
result_s = pd.DataFrame(columns = ['ITEMID','forecast_es','wape_es'])
pred_dates = pd.date_range(datetime.now(), datetime.now() + relativedelta(months=12), freq = 'MS')
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
      pred = model.predict(start = pred_dates[0],end =  pred_dates[-1])
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
      pred = model.predict(start = pred_dates[0],end =  pred_dates[-1])
      result_s.loc[counter, 'forecast_es'] = pred.values
      result_s.loc[counter,'wape_es'] = wape(test.values,for_wape.values)      

# COMMAND ----------

# для itemid с историей продаж <= 5 месяцев пытаться строить модели очевидно не имеет большого смысла ибо даже не на чем валидироваться и тестировать
# выход - подставляем средние за известный период
residuals = df[['ITEMID','salesQ','timestamp']]
residuals = df.groupby('ITEMID').filter(lambda x: len(x)<=5)
residuals_res = pd.DataFrame(columns=['ITEMID','forecast'])

for counter, i in enumerate(residuals.ITEMID.value_counts().index):
  residuals_res.loc[counter,'ITEMID'] = i
  residuals_res.loc[counter,'forecast'] = 12*[residuals[residuals.ITEMID==i].salesQ.mean()]
  
residuals_res = residuals_res.reset_index(drop=True)

# COMMAND ----------

# профет, сезонность и тренд мультидитивны, праздники - нг и 8 марта, если данных более чем за год период сезонности - год, иначе - количесвто известных месяцев продаж
prophet = df[['ITEMID','salesQ','timestamp']]
prophet = prophet.groupby('ITEMID').filter(lambda x: len(x)>5)
very_future = pd.DataFrame((
pd.date_range(datetime.now(), datetime.now() + relativedelta(months=12), freq = 'MS')
),columns = ['ds'])
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
print('prophet_ok')

# COMMAND ----------

# выбор лучшей модели между хольт-винтерсом и профетом, добавление средних
forecasting = result_s.set_index('ITEMID').join(result.set_index('ITEMID')).reset_index()
forecasting['forecast'] = 0
forecasting = forecasting.astype('object')

for index, content in forecasting.iterrows():
  if forecasting.loc[index,'wape_prophet'] < forecasting.loc[index,'wape_es']:
    forecasting.loc[index,'forecast'] = forecasting.loc[index,'forecast_prophet']
  else:
    forecasting.loc[index,'forecast'] = forecasting.loc[index,'forecast_es']
    
forecasting = forecasting.drop(['wape_prophet','wape_es','forecast_es','forecast_prophet'],1)

for index, content in forecasting.iterrows():
  forecasting['forecast'] = forecasting['forecast'].apply(lambda x : np.rint(x))
  
forecasting = pd.concat((forecasting,residuals_res))

norm_res = pd.DataFrame(columns = ['ITEMID','forecast','date'], index = range(0,12*8600))
for index , content in forecasting.iterrows():
    helper = pd.DataFrame(columns = ['ITEMID','forecast','date'], index = range(0,12))
    for counter, i in enumerate(content['forecast']):
      helper.loc[counter,'ITEMID'] = content['ITEMID']
      helper.loc[counter,'forecast'] = i
      helper.loc[counter,'date'] = pred_dates[counter]
    helper = helper.sort_values(by=['ITEMID','date'])
    norm_res = pd.concat((helper.dropna(),norm_res))
    norm_res = norm_res.reset_index(drop=True)
    helper = pd.DataFrame()
forecasting = norm_res.copy().dropna()

# COMMAND ----------

# запись данных в даталейк
forecasting.to_parquet('data/forecast.parquet', index=False)

# COMMAND ----------

# MAGIC %sh
# MAGIC 
# MAGIC ./azcopy copy data/forecast.parquet "https://alkordatalake.blob.core.windows.net/letoile/raw/ml/forecast.parquet?sv=2020-02-10&st=2020-12-31T21%3A00%3A00Z&se=2050-03-31T20%3A59%3A00Z&sr=c&sp=racwdlme&sig=69jYjI73deMsJohT8L9aDGjPE3k9ohHjdzmPPPkTD1s%3D"

# COMMAND ----------

# MAGIC %sh
# MAGIC ./azcopy remove "https://alkordatalake.dfs.core.windows.net/letoile/_%24azuretmpfolder%24?sv=2020-02-10&st=2020-12-31T21%3A00%3A00Z&se=2050-03-31T20%3A59%3A00Z&sr=c&sp=racwdlme&sig=69jYjI73deMsJohT8L9aDGjPE3k9ohHjdzmPPPkTD1s%3D" --recursive --trusted-microsoft-suffixes= --log-level=INFO;
# MAGIC 
# MAGIC ./azcopy remove "https://alkordatalake.dfs.core.windows.net/letoile/raw/ml/SalesLetoil?sv=2020-02-10&st=2020-12-31T21%3A00%3A00Z&se=2050-03-31T20%3A59%3A00Z&sr=c&sp=racwdlme&sig=69jYjI73deMsJohT8L9aDGjPE3k9ohHjdzmPPPkTD1s%3D" --recursive --trusted-microsoft-suffixes= --log-level=INFO;