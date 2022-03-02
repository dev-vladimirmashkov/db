# Databricks notebook source
# General imports
import numpy as np
import pandas as pd
import os, sys, gc, time, warnings, pickle, psutil, random
 
# custom imports
from multiprocessing import Pool        # Multiprocess Runs
from functools import partial
from typing import Union
 
warnings.filterwarnings('ignore')  

  
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
from scipy.optimize import minimize
from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt

# COMMAND ----------

print(pd.__version__)
print(np.__version__)

# COMMAND ----------

# функция WAPE
def wape(y_pred, y_true):
    res = np.sum(np.abs(y_true - y_pred)) / np.abs(np.sum(y_true)) * 100
    return res

# COMMAND ----------

########################### Helpers
#################################################################################
## Seeder
# :seed to make all processes deterministic     # type: int
def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
 
    
## Multiprocess Runs
def df_parallelize_run(func, t_split):
    num_cores = np.min([N_CORES,len(t_split)])
    pool = Pool(num_cores)
    df = pd.concat(pool.map(partial(func, base_test=base_test, TARGET=TARGET), t_split), axis=1)
    pool.close()
    pool.join()
    return df

# COMMAND ----------

########################### Helper to load data by store ID
#################################################################################
# Read data
def get_data_by_store(store):
    
    # Read and contact basic feature
    df = pd.concat([pd.read_pickle(BASE),
                    pd.read_pickle(PRICE).iloc[:,2:],
                    pd.read_pickle(CALENDAR).iloc[:,2:]],
                    axis=1)
    
    # Leave only relevant store
    df = df[df['store_id']==store]
 
    # With memory limits we have to read 
    # lags and mean encoding features
    # separately and drop items that we don't need.
    # As our Features Grids are aligned 
    # we can use index to keep only necessary rows
    # Alignment is good for us as concat uses less memory than merge.
    df2 = pd.read_pickle(MEAN_ENC)[mean_features]
    df2 = df2[df2.index.isin(df.index)]
    
    df3 = pd.read_pickle(LAGS).iloc[:,3:]
    df3 = df3[df3.index.isin(df.index)]
    
    df = pd.concat([df, df2], axis=1)
    del df2 # to not reach memory limit 
    
    df = pd.concat([df, df3], axis=1)
    del df3 # to not reach memory limit 
    
    # Create features list
    features = [col for col in list(df) if col not in remove_features]
    df = df[['id','d',TARGET]+features]
    
    # Skipping first n rows
    df = df[df['d']>=START_TRAIN].reset_index(drop=True)
    
    return df, features
 
# Recombine Test set after training
def get_base_test():
    base_test = pd.DataFrame()
 
    for store_id in STORES_IDS:
        temp_df = pd.read_pickle('test_'+store_id+'.pkl')
        temp_df['store_id'] = store_id
        base_test = pd.concat([base_test, temp_df]).reset_index(drop=True)
    
    return base_test
 
 
########################### Helper to make dynamic rolling lags
#################################################################################
def make_lag(LAG_DAY):
    lag_df = base_test[['id','d',TARGET]]
    col_name = 'sales_lag_'+str(LAG_DAY)
    lag_df[col_name] = lag_df.groupby(['id'])[TARGET].transform(lambda x: x.shift(LAG_DAY)).astype(np.float16)
    return lag_df[[col_name]]
 
 
def make_lag_roll(LAG_DAY):
    shift_day = LAG_DAY[0]
    roll_wind = LAG_DAY[1]
    lag_df = base_test[['id','d',TARGET]]
    col_name = 'rolling_mean_tmp_'+str(shift_day)+'_'+str(roll_wind)
    lag_df[col_name] = lag_df.groupby(['id'])[TARGET].transform(lambda x: x.shift(shift_day).rolling(roll_wind).mean())
    return lag_df[[col_name]]

# COMMAND ----------

class HoltWinters:
    
    """
    Holt-Winters model with the anomalies detection using Brutlag method
    
    # series - initial time series
    # slen - length of a season
    # alpha, beta, gamma - Holt-Winters model coefficients
    # n_preds - predictions horizon
    # scaling_factor - sets the width of the confidence interval by Brutlag (usually takes values from 2 to 3)
    
    """
    
    
    def __init__(self, series, slen, alpha, beta, gamma, n_preds, scaling_factor=1.96):
        self.series = series
        self.slen = slen
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.n_preds = n_preds
        self.scaling_factor = scaling_factor
        
        
    def initial_trend(self):
        sum = 0.0
        for i in range(self.slen):
            sum += float(self.series[i+self.slen] - self.series[i]) / self.slen
        return sum / self.slen  
    
    def initial_seasonal_components(self):
        seasonals = {}
        season_averages = []
        n_seasons = int(len(self.series)/self.slen)
        # let's calculate season averages
        for j in range(n_seasons):
            season_averages.append(sum(self.series[self.slen*j:self.slen*j+self.slen])/float(self.slen))
        # let's calculate initial values
        for i in range(self.slen):
            sum_of_vals_over_avg = 0.0
            for j in range(n_seasons):
                sum_of_vals_over_avg += self.series[self.slen*j+i]-season_averages[j]
            seasonals[i] = sum_of_vals_over_avg/n_seasons
        return seasonals   

          
    def triple_exponential_smoothing(self):
        self.result = []
        self.Smooth = []
        self.Season = []
        self.Trend = []
        self.PredictedDeviation = []
        self.UpperBond = []
        self.LowerBond = []
        
        seasonals = self.initial_seasonal_components()
        
        for i in range(len(self.series)+self.n_preds):
            if i == 0: # components initialization
                smooth = self.series[0]
                trend = self.initial_trend()
                self.result.append(self.series[0])
                self.Smooth.append(smooth)
                self.Trend.append(trend)
                self.Season.append(seasonals[i%self.slen])
                
                self.PredictedDeviation.append(0)
                
                self.UpperBond.append(self.result[0] + 
                                      self.scaling_factor * 
                                      self.PredictedDeviation[0])
                
                self.LowerBond.append(self.result[0] - 
                                      self.scaling_factor * 
                                      self.PredictedDeviation[0])
                continue
                
            if i >= len(self.series): # predicting
                m = i - len(self.series) + 1
                self.result.append((smooth + m*trend) + seasonals[i%self.slen])
                
                # when predicting we increase uncertainty on each step
                self.PredictedDeviation.append(self.PredictedDeviation[-1]*1.01) 
                
            else:
                val = self.series[i]
                last_smooth, smooth = smooth, self.alpha*(val-seasonals[i%self.slen]) + (1-self.alpha)*(smooth+trend)
                trend = self.beta * (smooth-last_smooth) + (1-self.beta)*trend
                seasonals[i%self.slen] = self.gamma*(val-smooth) + (1-self.gamma)*seasonals[i%self.slen]
                self.result.append(smooth+trend+seasonals[i%self.slen])
                
                # Deviation is calculated according to Brutlag algorithm.
                self.PredictedDeviation.append(self.gamma * np.abs(self.series[i] - self.result[i]) 
                                               + (1-self.gamma)*self.PredictedDeviation[-1])
                     
            self.UpperBond.append(self.result[-1] + 
                                  self.scaling_factor * 
                                  self.PredictedDeviation[-1])

            self.LowerBond.append(self.result[-1] - 
                                  self.scaling_factor * 
                                  self.PredictedDeviation[-1])

            self.Smooth.append(smooth)
            self.Trend.append(trend)
            self.Season.append(seasonals[i%self.slen])

# COMMAND ----------

# новый англ вариант
def timeseriesCVscore(params, series, loss_function=mean_squared_error, slen=7):
    """
        Returns error on CV  
        
        params - vector of parameters for optimization
        series - dataset with timeseries
        slen - season length for Holt-Winters model
    """
    # errors array
    errors = []
    
    values = series.values
    alpha, beta, gamma = params
    
    # set the number of folds for cross-validation
    tscv = TimeSeriesSplit(n_splits=7) 
    
    # iterating over folds, train model on each, forecast and calculate error
    for i, (train, test) in enumerate(tscv.split(values)):
        if i < 5:
            continue
        model = HoltWinters(series=values[train], slen=slen, 
                            alpha=alpha, beta=beta, gamma=gamma, n_preds=len(test))
        model.triple_exponential_smoothing()
        
        predictions = model.result[-len(test):]
        actual = values[test]
        error = loss_function(predictions, actual)
        errors.append(error)
        
    return np.mean(np.array(errors))

# COMMAND ----------

# MAGIC %sh
# MAGIC wget -O azcopy.tar.gz https://aka.ms/downloadazcopy-v10-linux
# MAGIC tar -xf azcopy.tar.gz
# MAGIC cp "$(dirname "$(find . -path ./azcopy_linux\* -type f| tail -1)")"/azcopy azcopy
# MAGIC mkdir data
# MAGIC mkdir data1
# MAGIC mkdir test
# MAGIC mkdir models
# MAGIC   
# MAGIC ./azcopy copy "https://bricksresult.blob.core.windows.net/data/20200308wall/*?st=2020-03-04T06%3A58%3A48Z&se=2020-12-05T06%3A58%3A00Z&sp=racwdl&sv=2018-03-28&sr=c&sig=LFmu8D%2FVsGZfEj%2F45W6nldF2lvq2mQr9kFl6AblqUN8%3D" "data/" 

# COMMAND ----------

########################### Vars
#################################################################################
VER = "1913_14_0"                          # Our model version
SEED = 42                        # We want all things
seed_everything(SEED)            # to be as deterministic 
# lgb_params['seed'] = SEED        # as possible
N_CORES = psutil.cpu_count()     # Available CPU cores
VERBOSE   = False 
 
#LIMITS and const
TARGET      = 'sales'            # Our target
START_TRAIN = 0                  # We can skip some rows (Nans/faster training)
END_TRAIN   = 1913               # End day of our train set
P_HORIZON   = 28                 # Prediction horizon
USE_AUX     = False               # Use or not pretrained models
 
#FEATURES to remove
## These features lead to overfit
## or values not present in test set
remove_features = ['id','state_id','store_id',
                   'date','wm_yr_wk','d',TARGET]
mean_features   = ['enc_cat_id_mean','enc_cat_id_std',
                   'enc_dept_id_mean','enc_dept_id_std',
                   'enc_item_id_mean','enc_item_id_std'] 
 
#PATHS for Features
ORIGINAL = 'data/'
BASE     = 'data/grid_part_1.pkl'
PRICE    = 'data/grid_part_2.pkl'
CALENDAR = 'data/grid_part_3.pkl'
LAGS     = 'data/lags_df_28.pkl'
MEAN_ENC = 'data/mean_encoding_df.pkl'
 
 
# AUX(pretrained) Models paths
AUX_MODELS = 'models/'
 
 
#STORES ids
STORES_IDS = pd.read_csv(ORIGINAL+'sales_train_validation.csv')['store_id']
STORES_IDS = list(STORES_IDS.unique())
#DEPTS ids
DEPTS_IDS = pd.read_csv(ORIGINAL+'sales_train_validation.csv')['dept_id']
DEPTS_IDS = list(DEPTS_IDS.unique())  
 
#SPLITS for lags creation
SHIFT_DAY  = 28
N_LAGS     = 15
LAGS_SPLIT = [col for col in range(SHIFT_DAY,SHIFT_DAY+N_LAGS)]
ROLS_SPLIT = []
for i in [1,7,14]:
    for j in [7,14,30,60]:
        ROLS_SPLIT.append([i,j])

# COMMAND ----------

df = pd.concat([pd.read_pickle(BASE),
                pd.read_pickle(PRICE).iloc[:,2:],
                pd.read_pickle(CALENDAR).iloc[:,2:]],
                axis=1)
df.head()

# COMMAND ----------

df['OOS'] = 0
df.loc[df.sales == 0,'OOS'] = 1

# COMMAND ----------

df[['sales','release', 'OOS']]

# COMMAND ----------

train_df = pd.read_csv(ORIGINAL+'sales_train_validation.csv')
prices_df = pd.read_csv(ORIGINAL+'sell_prices.csv')
calendar_df = pd.read_csv(ORIGINAL+'calendar.csv')

# COMMAND ----------

########################## Make Grid
#################################################################################
print('Create Grid')

# We can tranform horizontal representation 
# to vertical "view"
# Our "index" will be 'id','item_id','dept_id','cat_id','store_id','state_id'
# and labels are 'd_' coulmns

index_columns = ['id','item_id','dept_id','cat_id','store_id','state_id']
grid_df = pd.melt(train_df, 
                  id_vars = index_columns, 
                  var_name = 'd', 
                  value_name = TARGET)

# If we look on train_df we se that 
# we don't have a lot of traning rows
# but each day can provide more train data
print('Train rows:', len(train_df), len(grid_df))

# To be able to make predictions
# we need to add "test set" to our grid
add_grid = pd.DataFrame()
for i in range(1,29):
    temp_df = train_df[index_columns]
    temp_df = temp_df.drop_duplicates()
    temp_df['d'] = 'd_'+ str(END_TRAIN+i)
    temp_df[TARGET] = np.nan
    add_grid = pd.concat([add_grid,temp_df])

grid_df = pd.concat([grid_df,add_grid])
grid_df = grid_df.reset_index(drop=True)

# Remove some temoprary DFs
del temp_df, add_grid

# We will not need original train_df
# anymore and can remove it
del train_df

# You don't have to use df = df construction
# you can use inplace=True instead.
# like this
# grid_df.reset_index(drop=True, inplace=True)

# Let's check our memory usage
print("{:>20}: {:>8}".format('Original grid_df',sizeof_fmt(grid_df.memory_usage(index=True).sum())))

# We can free some memory 
# by converting "strings" to categorical
# it will not affect merging and 
# we will not lose any valuable data
for col in index_columns:
    grid_df[col] = grid_df[col].astype('category')

# Let's check again memory usage
print("{:>20}: {:>8}".format('Reduced grid_df',sizeof_fmt(grid_df.memory_usage(index=True).sum())))

# COMMAND ----------



# COMMAND ----------

########################### Product Release date
#################################################################################
print('Release week')

# It seems that leadings zero values
# in each train_df item row
# are not real 0 sales but mean
# absence for the item in the store
# we can safe some memory by removing
# such zeros

# Prices are set by week
# so it we will have not very accurate release week 
release_df = prices_df.groupby(['store_id','item_id'])['wm_yr_wk'].agg(['min']).reset_index()
release_df.columns = ['store_id','item_id','release']

# Now we can merge release_df
grid_df = merge_by_concat(grid_df, release_df, ['store_id','item_id'])
# del release_df

# We want to remove some "zeros" rows
# from grid_df 
# to do it we need wm_yr_wk column
# let's merge partly calendar_df to have it
grid_df = merge_by_concat(grid_df, calendar_df[['wm_yr_wk','d']], ['d'])
                      
# Now we can cutoff some rows 
# and safe memory 
grid_df = grid_df[grid_df['wm_yr_wk']>=grid_df['release']]
grid_df = grid_df.reset_index(drop=True)

# Let's check our memory usage
print("{:>20}: {:>8}".format('Original grid_df',sizeof_fmt(grid_df.memory_usage(index=True).sum())))

# Should we keep release week 
# as one of the features?
# Only good CV can give the answer.
# Let's minify the release values.
# Min transformation will not help here 
# as int16 -> Integer (-32768 to 32767)
# and our grid_df['release'].max() serves for int16
# but we have have an idea how to transform 
# other columns in case we will need it
grid_df['release'] = grid_df['release'] - grid_df['release'].min()
grid_df['release'] = grid_df['release'].astype(np.int16)

# COMMAND ----------

prices_df

# COMMAND ----------

df.sample(5)

# COMMAND ----------



# COMMAND ----------

prices_df.tail()

# COMMAND ----------

i = 0
df7 = []
for store_id in STORES_IDS:
  for dept_id in DEPTS_IDS:
    print(store_id)
    print(dept_id)
    grid_df, features_columns = get_data_by_store_dept(store_id, dept_id)
    grid_df['sell_price'] = grid_df['sell_price'].astype(np.float64).round(2)
    grid_df['money'] = grid_df.sales * grid_df.sell_price
    df3 = grid_df.groupby('d')['money'].sum().reset_index()
    data = df3.money[:-56]
    # инициализируем значения параметров
    x = [0, 0, 0] 

    # Минимизируем функцию потерь с ограничениями на параметры
    opt = minimize(timeseriesCVscore, x0=x, 
                   args=(data, mean_squared_error, 7), 
                   method="TNC", bounds = ((0, 1), (0, 1), (0, 1))
                  )

    # Из оптимизатора берем оптимальное значение параметров
    alpha_final, beta_final, gamma_final = opt.x
    print(alpha_final, beta_final, gamma_final)
#     недельные циклы прогнозируем и записываем в дф3
    data = df3.money[:-28]
    model = HoltWinters(data[:-28], slen = 7, alpha = alpha_final, beta = beta_final, gamma = gamma_final, n_preds = 56, scaling_factor = 3) 
    model.triple_exponential_smoothing()
    print(wape(model.result[-56:-28],data[-28:]).round(2))
    w0 = wape(model.result[-56:-28],data[-28:]).round(2)
    df3['Dayforecast'] = np.nan  
    model = HoltWinters(data, slen = 7, alpha = alpha_final, beta = beta_final, gamma = gamma_final, n_preds = 28, scaling_factor = 3) 
    model.triple_exponential_smoothing()
    df3.iloc[:,-1] = model.result
    df3['WapeDay'] = w0
    for l in range(5):
      n = (l +2) * (-28)
      data = df3.money[:n]
      model = HoltWinters(data, slen = 7, alpha = alpha_final, beta = beta_final, gamma = gamma_final, n_preds = 28, scaling_factor = 3) #n_preds=28 для графика
      model.triple_exponential_smoothing()
      df3.iloc[n:(n+28),-2] = model.result[-28:]
#     тренд по неделям записываем в дф3
    df3['week'] = (df3.d-3)//7
    df4 = df3.groupby('week')['money'].sum().reset_index()
    data = df4.money[1:-8]
    # инициализируем значения параметров
    x = [0, 0, 0] 

    # Минимизируем функцию потерь с ограничениями на параметры
    opt = minimize(timeseriesCVscore, x0=x, 
                   args=(data, mean_squared_error, 52), 
                   method="TNC", bounds = ((0, 1), (0, 1), (0, 1))
                  )

    # Из оптимизатора берем оптимальное значение параметров
    alpha_final, beta_final, gamma_final = opt.x
    print(alpha_final, beta_final, gamma_final)
    data = df4.money[1:-4].reset_index(drop=True)
    model = HoltWinters(data[:-4], slen = 52, alpha = alpha_final, beta = beta_final, gamma = gamma_final, n_preds = 8, scaling_factor = 3) 
    model.triple_exponential_smoothing()
    print(wape(model.result[-8:-4],data[-4:]).round(2))
    w1 = wape(model.result[-8:-4],data[-4:]).round(2)
    model = HoltWinters(data, slen = 52, alpha = alpha_final, beta = beta_final, gamma = gamma_final, n_preds = 4, scaling_factor = 3) 
    model.triple_exponential_smoothing()
    df4['Weekforecast'] = np.nan  
    df4.iloc[1:,-1] = model.result
    df4.iloc[0,-1] = df4.iloc[1,-1]
    df4['WapeWeek'] = w1
    for l in range(5):
      n = (l +2) * (-4)
      data = df4.money[1:n].reset_index(drop=True)
      model = HoltWinters(data, slen = 52, alpha = alpha_final, beta = beta_final, gamma = gamma_final, n_preds = 4, scaling_factor = 3) #n_preds=4 для графика
      model.triple_exponential_smoothing()
      df4.iloc[n:(n+4),-2] = model.result[-4:]
#             print(df4.iloc[n:(n+4),-1])
    df3 = df3.merge(df4, how='left', on=['week'])
    df3.columns = ['d', 'money_day_fact', 'DayforecastDept', 'WapeDayDept', 'week', 'money_mean_week', 'WeekforecastDept', 'WapeWeekDept']
    df5 = df3[['d', 'DayforecastDept', 'WapeDayDept', 'WeekforecastDept', 'WapeWeekDept']]
    grid_df = grid_df.merge(df5, how='left', on=['d'])    
    if i == 0:
      df7 = grid_df[['id','d', 'DayforecastDept', 'WapeDayDept', 'WeekforecastDept','WapeWeekDept']]
    else:
      df7 = pd.concat([df7, grid_df[['id','d', 'DayforecastDept', 'WapeDayDept', 'WeekforecastDept','WapeWeekDept']]])
    del grid_df, df3, df4, df5
    gc.collect()
    i +=1

# COMMAND ----------

df7.to_pickle('grid_part_732_2dept.pkl')

# COMMAND ----------



# COMMAND ----------

# MAGIC %sh
# MAGIC  
# MAGIC ./azcopy copy "grid_part_732_2dept.pkl" "https://bricksresult.blob.core.windows.net/data/20200308wall/grid_part_732_2dept.pkl?st=2020-03-04T06%3A58%3A48Z&se=2020-12-05T06%3A58%3A00Z&sp=racwdl&sv=2018-03-28&sr=c&sig=LFmu8D%2FVsGZfEj%2F45W6nldF2lvq2mQr9kFl6AblqUN8%3D" 

# COMMAND ----------

df7