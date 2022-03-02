# Databricks notebook source
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

df = read_parquet_folder_as_pandas('data')
df.drop(df[df.salesQ==0].index,inplace=True)
df['timestamp'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str),
                                              format='%Y-%m')

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

# COMMAND ----------

df['month*price'] = df['month'] * df['price']
df['month*price_lag'] = df['month'] * df['price_lag']
df['price_lag*price'] = df['price_lag'] * df['price']
df['month*month'] = df['month'] * df['month']
df['price*price'] = df['price'] * df['price']
df['price_lag*price_lag'] = df['price_lag'] * df['price_lag']
df['wape_indicator'] = np.where(wape(df.price_lag , df.price)>75,1,0)



# COMMAND ----------

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

features = ['SalesQ_lag1', 'SalesM_lag4', 'rolling_mean_M1_3', 'rolling_std_M2_9', 'price', 'rolling_mean_M2_3', 'SalesM_lag2',  'rolling_std_M3_6', 'rolling_mean_M2_9', 'rolling_std_M2_2', 'rolling_std_M1_2', 'rolling_std_M3_2', 'rolling_mean_M1_6', 'rolling_std_M3_3', 'rolling_mean_M3_6', 'rolling_mean_M2_6','month*price_lag']

# COMMAND ----------

from sklearn.manifold import TSNE

viz = TSNE(n_components=2,verbose=1).fit_transform(df[features].fillna(0))

# COMMAND ----------

df['viz_x'] = viz[:,0]
df['viz_y'] = viz[:,1]

# COMMAND ----------

plt.figure(figsize=(20,10))
sns.scatterplot(x='viz_x',
               y='viz_y',
               hue='month',data=df)

# COMMAND ----------

plt.figure(figsize=(20,10))
sns.scatterplot(x='viz_x',
               y='viz_y',
               hue='year',data=df)

# COMMAND ----------


df_saved = spark.sql("select * from df")

# COMMAND ----------

