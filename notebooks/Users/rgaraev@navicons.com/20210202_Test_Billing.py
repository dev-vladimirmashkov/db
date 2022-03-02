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

# DBTITLE 1,INFO
# Для проверки биллинга взял начальный feature engineering из задачи предсказания товаров в чеке Летуаля.
# Данный джобс на кластере ML Single Node выполняется не более 7-8 минут
# Job_20210202_Test_Billing

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

