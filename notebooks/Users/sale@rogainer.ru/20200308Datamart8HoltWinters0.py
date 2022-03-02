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

!pip install lightgbm

# COMMAND ----------

import lightgbm as lgb
lgb_params = {
                    'boosting_type': 'gbdt',
                    'objective': 'tweedie',
                    'tweedie_variance_power': 1.1,
                    'metric': 'rmse',
                    'subsample': 0.5,
                    'subsample_freq': 1,
                    'learning_rate': 0.03,
                    'num_leaves': 2**11-1,
                    'min_data_in_leaf': 2**12-1,
                    'feature_fraction': 0.5,
                    'max_bin': 100,
                    'n_estimators': 2000,
                    'early_stopping_rounds': 500,
                    'boost_from_average': False,
                    'verbose': -1,
                } 

# COMMAND ----------

########################### Init Metric
########################### https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/133834
#################################################################################
class WRMSSEEvaluator(object):
 
    def __init__(self, train_df: pd.DataFrame, valid_df: pd.DataFrame, calendar: pd.DataFrame, prices: pd.DataFrame):
        train_y = train_df.loc[:, train_df.columns.str.startswith('d_')]
        train_target_columns = train_y.columns.tolist()
        weight_columns = train_y.iloc[:, -28:].columns.tolist()
 
        train_df['all_id'] = 0
 
        id_columns = train_df.loc[:, ~train_df.columns.str.startswith('d_')].columns.tolist()
        valid_target_columns = valid_df.loc[:, valid_df.columns.str.startswith('d_')].columns.tolist()
 
        if not all([c in valid_df.columns for c in id_columns]):
            valid_df = pd.concat([train_df[id_columns], valid_df], axis=1, sort=False)
 
        self.train_df = train_df.copy()
        self.valid_df = valid_df.copy()
        self.calendar = calendar.copy()
        self.prices = prices.copy()
 
        self.weight_columns = weight_columns
        self.id_columns = id_columns
        self.valid_target_columns = valid_target_columns
 
        weight_df = self.get_weight_df()
 
        self.group_ids = (
            'all_id',
            'state_id',
            'store_id',
            'cat_id',
            'dept_id',
            ['state_id', 'cat_id'],
            ['state_id', 'dept_id'],
            ['store_id', 'cat_id'],
            ['store_id', 'dept_id'],
            'item_id',
            ['item_id', 'state_id'],
            ['item_id', 'store_id']
        )
 
        for i, group_id in enumerate(self.group_ids):
            train_y = train_df.groupby(group_id)[train_target_columns].sum()
            scale = []
            for _, row in train_y.iterrows():
                series = row.values[np.argmax(row.values != 0):]
                scale.append(((series[1:] - series[:-1]) ** 2).mean())
            setattr(self, f'lv{i + 1}_scale', np.array(scale))
            setattr(self, f'lv{i + 1}_train_df', train_y)
            setattr(self, f'lv{i + 1}_valid_df', valid_df.groupby(group_id)[valid_target_columns].sum())
 
            lv_weight = weight_df.groupby(group_id)[weight_columns].sum().sum(axis=1)
            setattr(self, f'lv{i + 1}_weight', lv_weight / lv_weight.sum())
 
    def get_weight_df(self) -> pd.DataFrame:
        day_to_week = self.calendar.set_index('d')['wm_yr_wk'].to_dict()
        weight_df = self.train_df[['item_id', 'store_id'] + self.weight_columns].set_index(['item_id', 'store_id'])
        weight_df = weight_df.stack().reset_index().rename(columns={'level_2': 'd', 0: 'value'})
        weight_df['wm_yr_wk'] = weight_df['d'].map(day_to_week)
 
        weight_df = weight_df.merge(self.prices, how='left', on=['item_id', 'store_id', 'wm_yr_wk'])
        weight_df['value'] = weight_df['value'] * weight_df['sell_price']
        weight_df = weight_df.set_index(['item_id', 'store_id', 'd']).unstack(level=2)['value']
        weight_df = weight_df.loc[zip(self.train_df.item_id, self.train_df.store_id), :].reset_index(drop=True)
        weight_df = pd.concat([self.train_df[self.id_columns], weight_df], axis=1, sort=False)
        return weight_df
 
    def get_scale(self, valid_preds: pd.DataFrame, lv: int) -> pd.Series:
        return getattr(self, f'lv{lv}_scale')
        
    def rmsse(self, valid_preds: pd.DataFrame, lv: int) -> pd.Series:
        valid_y = getattr(self, f'lv{lv}_valid_df')
        score = ((valid_y - valid_preds) ** 2).mean(axis=1)
        scale = getattr(self, f'lv{lv}_scale')       
        return (score / scale).map(np.sqrt)
 
    def score(self, valid_preds: Union[pd.DataFrame, np.ndarray]) -> float:
        assert self.valid_df[self.valid_target_columns].shape == valid_preds.shape
 
        if isinstance(valid_preds, np.ndarray):
            valid_preds = pd.DataFrame(valid_preds, columns=self.valid_target_columns)
 
        valid_preds = pd.concat([self.valid_df[self.id_columns], valid_preds], axis=1, sort=False)
 
        all_scores = []
        for i, group_id in enumerate(self.group_ids):
            lv_scores = self.rmsse(valid_preds.groupby(group_id)[self.valid_target_columns].sum(), i + 1)
            weight = getattr(self, f'lv{i + 1}_weight')
            lv_scores = pd.concat([weight, lv_scores], axis=1, sort=False).prod(axis=1)
            all_scores.append(lv_scores.sum())
        if VERBOSE:
            print(np.round(all_scores,3))
        return np.mean(all_scores)
 
    def full_score(self, valid_preds: Union[pd.DataFrame, np.ndarray]) -> float:
        assert self.valid_df[self.valid_target_columns].shape == valid_preds.shape
 
        if isinstance(valid_preds, np.ndarray):
            valid_preds = pd.DataFrame(valid_preds, columns=self.valid_target_columns)
 
        valid_preds = pd.concat([self.valid_df[self.id_columns], valid_preds], axis=1, sort=False)
 
        all_scores = []
        for i, group_id in enumerate(self.group_ids):
            lv_scores = self.rmsse(valid_preds.groupby(group_id)[self.valid_target_columns].sum(), i + 1)
            weight = getattr(self, f'lv{i + 1}_weight')
            lv_scores = pd.concat([weight, lv_scores], axis=1, sort=False).prod(axis=1)
            all_scores.append(lv_scores.sum())
        print(np.round(all_scores,3))
        return np.mean(all_scores)
    
class WRMSSEForLightGBM(WRMSSEEvaluator):
 
    def feval(self, preds, dtrain):
        preds = preds.reshape(self.valid_df[self.valid_target_columns].shape, order='F') #.transpose()
        score = self.score(preds)
        return 'WRMSSE', score, False
    
    def full_feval(self, preds, dtrain):
        preds = preds.reshape(self.valid_df[self.valid_target_columns].shape, order='F') #.transpose()
        score = self.full_score(preds)
        return 'WRMSSE', score, False
    
########################### Lgb evaluators
#################################################################################
def get_evaluators(ids):
    prices = pd.read_csv('data/sell_prices.csv')
    calendar = pd.read_csv('data/calendar.csv')
    train_fold_df = pd.read_csv('data/sales_train_validation.csv')
#     train_fold_df = train_fold_df[train_fold_df['store_id'] == store_id].reset_index(drop=True)
    train_fold_df = train_fold_df[train_fold_df['id'].isin(ids)].reset_index(drop=True)
    lgb_evaluator = []
    temp_train = train_fold_df.iloc[:,:-28]
    temp_valid = train_fold_df.iloc[:, -28:]    
    lgb_evaluator = WRMSSEForLightGBM(temp_train, temp_valid, calendar, prices)
    
    del train_fold_df, temp_train, temp_valid, prices, calendar
    return lgb_evaluator

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
# MAGIC ./azcopy copy "https://bricksresult.blob.core.windows.net/data/20200308models/*?st=2020-03-04T06%3A58%3A48Z&se=2020-12-05T06%3A58%3A00Z&sp=racwdl&sv=2018-03-28&sr=c&sig=LFmu8D%2FVsGZfEj%2F45W6nldF2lvq2mQr9kFl6AblqUN8%3D" "models/"

# COMMAND ----------

########################### Vars
#################################################################################
VER = "1913_8"                          # Our model version
SEED = 42                        # We want all things
seed_everything(SEED)            # to be as deterministic 
lgb_params['seed'] = SEED        # as possible
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
 
 
#SPLITS for lags creation
SHIFT_DAY  = 28
N_LAGS     = 15
LAGS_SPLIT = [col for col in range(SHIFT_DAY,SHIFT_DAY+N_LAGS)]
ROLS_SPLIT = []
for i in [1,7,14]:
    for j in [7,14,30,60]:
        ROLS_SPLIT.append([i,j])

# COMMAND ----------

df = pd.read_csv(ORIGINAL+'sales_train_validation.csv')
df.head()

# COMMAND ----------

df1 = pd.concat([pd.read_pickle(BASE),
                    pd.read_pickle(PRICE).iloc[:,2:],
                    pd.read_pickle(CALENDAR).iloc[:,2:]],
                    axis=1)
df1.head()

# COMMAND ----------

del df, df1
gc.collect()

# COMMAND ----------



# COMMAND ----------

grid_df, features_columns = get_data_by_store(STORES_IDS[0])
grid_df.head()

# COMMAND ----------

grid_df.shape

# COMMAND ----------

len(grid_df['item_id'].unique())

# COMMAND ----------

df2 = grid_df[['id','d','sales','sell_price']].copy()
del grid_df
gc.collect()

# COMMAND ----------

prices = pd.read_csv(ORIGINAL+'sell_prices.csv')

# COMMAND ----------

prices[prices['item_id'] == 'HOBBIES_1_015'].head()

# COMMAND ----------

df2.head()

# COMMAND ----------

df2['sell_price'] = df2['sell_price'].astype(np.float64).round(2)
df2.head()

# COMMAND ----------



# COMMAND ----------

df2['money'] = df2.sales * df2.sell_price
df3 = df2.groupby('d')['money'].sum().reset_index()
df3.head()

# COMMAND ----------

df3

# COMMAND ----------

df3[-30:]

# COMMAND ----------

df2

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

# Кросс-валидация на временных рядах
from sklearn.model_selection import TimeSeriesSplit

def timeseriesCVscore(x):
    # вектор ошибок
    errors = []

    values = data.values
    alpha, beta, gamma = x

    # задаём число фолдов для кросс-валидации
    tscv = TimeSeriesSplit(n_splits=5) 

    # идем по фолдам, на каждом обучаем модель, строим прогноз на отложенной выборке и считаем ошибку
    for i, (train, test) in enumerate(tscv.split(values)):
        if i < 4:
            continue
        model = HoltWinters(series=values[train], slen = 365, alpha=alpha, beta=beta, gamma=gamma, n_preds=len(test))
        model.triple_exponential_smoothing()

        predictions = model.result[-len(test):]
        actual = values[test]
        error = mean_squared_error(predictions, actual)
        errors.append(error)

    # Возвращаем средний квадрат ошибки по вектору ошибок 
    return np.mean(np.array(errors))

# COMMAND ----------

df3.head()

# COMMAND ----------

data = df3.money[:-56]
# инициализируем значения параметров
x = [0.5, 0, 0.5] 

# Минимизируем функцию потерь с ограничениями на параметры
opt = minimize(timeseriesCVscore, x0=x, method="TNC", bounds = ((0, 1), (0, 0), (0, 1)))

# Из оптимизатора берем оптимальное значение параметров
alpha_final, beta_final, gamma_final = opt.x
print(alpha_final, beta_final, gamma_final)

# COMMAND ----------

data = df3.money[:-28]
model = HoltWinters(data[:-28], slen = 365, alpha = alpha_final, beta = beta_final, gamma = gamma_final, n_preds = 28, scaling_factor = 2.56)
model.triple_exponential_smoothing()

# COMMAND ----------

def plotHoltWinters():
    Anomalies = np.array([np.NaN]*len(data))
    Anomalies[data.values<model.LowerBond] = data.values[data.values<model.LowerBond]
    Anomalies[data.values>model.UpperBond] = data.values[data.values>model.UpperBond]
    plt.figure(figsize=(25, 10))
    plt.plot(model.result, label = "Model")
    plt.plot(model.UpperBond, "r--", alpha=0.5, label = "Up/Low confidence")
    plt.plot(model.LowerBond, "r--", alpha=0.5)
    plt.fill_between(x=range(0,len(model.result)), y1=model.UpperBond, y2=model.LowerBond, alpha=0.5, color = "grey")
    plt.plot(data.values, label = "Actual")
    plt.plot(Anomalies, "o", markersize=10, label = "Anomalies")
    plt.axvspan(len(data)-28, len(data), alpha=0.5, color='lightgrey')
    plt.grid(True)
    plt.axis('tight')
#     plt.title(" RMSE {} % sales".format(wape(model.result[-11:],TotalSalesConstP.TRDFactPC[-11:]).round(2)))
    plt.legend(loc="best", fontsize=13)
    plt.xlim(1750,1913)

plotHoltWinters()

# COMMAND ----------

df2['week'] = (df2.d-3)//7
df2

# COMMAND ----------

df4 = df2.groupby('week')['money'].sum().reset_index()
df4.tail(10)

# COMMAND ----------

(1913-2)%7

# COMMAND ----------

def timeseriesCVscore(x):
    # вектор ошибок
    errors = []

    values = data.values
    alpha, beta, gamma = x

    # задаём число фолдов для кросс-валидации
    tscv = TimeSeriesSplit(n_splits=5) 

    # идем по фолдам, на каждом обучаем модель, строим прогноз на отложенной выборке и считаем ошибку
    for i, (train, test) in enumerate(tscv.split(values)):
        if i < 4:
            continue
        model = HoltWinters(series=values[train], slen = 52, alpha=alpha, beta=beta, gamma=gamma, n_preds=len(test))
        model.triple_exponential_smoothing()

        predictions = model.result[-len(test):]
        actual = values[test]
        error = mean_squared_error(predictions, actual)
        errors.append(error)

    # Возвращаем средний квадрат ошибки по вектору ошибок 
    return np.mean(np.array(errors))

# COMMAND ----------

data = df4.money[1:-8]
# инициализируем значения параметров
x = [0, 0, 0] 

# Минимизируем функцию потерь с ограничениями на параметры
opt = minimize(timeseriesCVscore, x0=x, method="TNC", bounds = ((0, 1), (0, 1), (0, 1)))

# Из оптимизатора берем оптимальное значение параметров
alpha_final, beta_final, gamma_final = opt.x
print(alpha_final, beta_final, gamma_final)

# COMMAND ----------

# новый англ вариант
def timeseriesCVscore(params, series, loss_function=mean_squared_error, slen=52):
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

# COMMAND ----------

data = df4.money[1:-4].reset_index(drop=True)
model = HoltWinters(data[:-4], slen = 52, alpha = alpha_final, beta = beta_final, gamma = gamma_final, n_preds = 8, scaling_factor = 3) #n_preds=4 для графика
model.triple_exponential_smoothing()
df4['Weekforecast'] = np.nan  
df4.iloc[1:,-1] = model.result

# COMMAND ----------



# COMMAND ----------

df4.tail(10)

# COMMAND ----------

def plotHoltWinters():
    Anomalies = np.array([np.NaN]*len(data))
    Anomalies[data.values<model.LowerBond] = data.values[data.values<model.LowerBond]
    Anomalies[data.values>model.UpperBond] = data.values[data.values>model.UpperBond]
    plt.figure(figsize=(25, 10))
    plt.plot(model.result, label = "Model")
    plt.plot(model.UpperBond, "r--", alpha=0.5, label = "Up/Low confidence")
    plt.plot(model.LowerBond, "r--", alpha=0.5)
    plt.fill_between(x=range(0,len(model.result)), y1=model.UpperBond, y2=model.LowerBond, alpha=0.5, color = "grey")
    plt.plot(data.values, label = "Actual")
    plt.plot(Anomalies, "o", markersize=10, label = "Anomalies")
    plt.axvspan(len(data)-4, len(data), alpha=0.5, color='lightgrey')
    plt.grid(True)
    plt.axis('tight')
    plt.title(" WAPE {} % sales".format(wape(model.result[-4:],data[-4:]).round(2)))
    plt.legend(loc="best", fontsize=13)
#     plt.xlim(274-53-4-8,274)

plotHoltWinters()

# COMMAND ----------

wape(model.result[-4:],data[-4:]).round(2)

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
    print(slen)
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

  
  
data = df3.money[:-56]
# инициализируем значения параметров
x = [0, 0, 0] 

# Минимизируем функцию потерь с ограничениями на параметры
opt = minimize(timeseriesCVscore, x0=x, 
               args=(data, mean_squared_error,14), 
               method="TNC", bounds = ((0, 1), (0, 1), (0, 1))
              )

# Из оптимизатора берем оптимальное значение параметров
alpha_final, beta_final, gamma_final = opt.x
print(alpha_final, beta_final, gamma_final)

# COMMAND ----------

data = df3.money[:-28]
model = HoltWinters(data[:-28], slen = 7, alpha = alpha_final, beta = beta_final, gamma = gamma_final, n_preds = 56, scaling_factor = 2.56) #n_preds=28 для графика
model.triple_exponential_smoothing()
df3['Dayforecast'] = np.nan  
df3.iloc[:,-1] = model.result

# COMMAND ----------

print(wape(model.result[-56:-28],data[-28:]).round(2))

# COMMAND ----------

df3.tail(30)

# COMMAND ----------

def plotHoltWinters():
    Anomalies = np.array([np.NaN]*len(data))
    Anomalies[data.values<model.LowerBond] = data.values[data.values<model.LowerBond]
    Anomalies[data.values>model.UpperBond] = data.values[data.values>model.UpperBond]
    plt.figure(figsize=(25, 10))
    plt.plot(model.result, label = "Model")
    plt.plot(model.UpperBond, "r--", alpha=0.5, label = "Up/Low confidence")
    plt.plot(model.LowerBond, "r--", alpha=0.5)
    plt.fill_between(x=range(0,len(model.result)), y1=model.UpperBond, y2=model.LowerBond, alpha=0.5, color = "grey")
    plt.plot(data.values, label = "Actual")
    plt.plot(Anomalies, "o", markersize=10, label = "Anomalies")
    plt.axvspan(len(data)-28, len(data), alpha=0.5, color='lightgrey')
    plt.grid(True)
    plt.axis('tight')
    plt.title(" WAPE {} % sales".format(wape(model.result[-56:-28],data[-28:]).round(2)))
    plt.legend(loc="best", fontsize=13)
    plt.xlim(1750,1913)
    plt.ylim(0,25000)

plotHoltWinters()

# COMMAND ----------

df3['week'] = (df3.d-3)//7
df3.head()

# COMMAND ----------

df4.head()

# COMMAND ----------

df3 = df3.merge(df4, how='left', on=['week'])
df3.head()

# COMMAND ----------

df3.tail(30)

# COMMAND ----------

df3.columns

# COMMAND ----------

df3.columns = ['d', 'money_day_fact', 'Dayforecast', 'week', 'money_mean_week', 'Weekforecast']

# COMMAND ----------

df3.money_mean_week = df3.money_mean_week/7
df3.Weekforecast = df3.Weekforecast/7

# COMMAND ----------

plt.figure(figsize=(25, 10))
plt.plot(df3.d, df3.money_day_fact)
plt.plot(df3.d, df3.Dayforecast)
plt.plot(df3.d, df3.money_mean_week)
plt.plot(df3.d, df3.Weekforecast)
plt.legend(loc="best", fontsize=13)
plt.xlim(1750,1941)
plt.ylim(0,25000)

# COMMAND ----------



# COMMAND ----------

import lightgbm as lgb
lgbm = lgb.LGBMRegressor(
                        n_estimators=800,
                        learning_rate=0.01,
                        feature_fraction=0.7,
                        subsample=0.4,
                        num_leaves=40,
                        metric='mae')

# параметры lightgbm
kwargs = {'early_stopping_rounds':50,'verbose':10}

# функция кросс-валидации по времени
def LGBMTimeSeriesCV(X_train, y_train, number_folds, model, metrics, kwargs={}):
    print('Size train set: {}'.format(X_train.shape))

    k = int(np.floor(float(X_train.shape[0]) / number_folds))
    print('Size of each fold: {}'.format(k))

    errors = np.zeros(number_folds-1)

    # loop from the first 2 folds to the total number of folds    
    for i in range(2, number_folds + 1):
        print('')
        split = float(i-1)/i
        print('Splitting the first ' + str(i) + ' chunks at ' + str(i-1) + '/' + str(i) )

        X = X_train[:(k*i)]
        y = y_train[:(k*i)]
        print('Size of train + test: {}'.format(X.shape)) # the size of the dataframe is going to be k*i

        index = int(np.floor(X.shape[0] * split))

        # folds used to train the model        
        X_trainFolds = X[:index]        
        y_trainFolds = y[:index]

        # fold used to test the model
        X_testFold = X[(index + 1):]
        y_testFold = y[(index + 1):]

        model.fit(X_trainFolds, y_trainFolds, **kwargs, eval_set=[(X_testFold, y_testFold)])
        errors[i-2] = metrics(model.predict(X_testFold), y_testFold)
        print(errors[i-2])

    # the function returns the mean of the errors on the n-1 folds    
    return errors.mean()

# COMMAND ----------

df3.head()

# COMMAND ----------

df3.tail()

# COMMAND ----------

# готовим данные без лагов целевой
def prepareData(data, test_size=28):

    data = data.copy()
    data["y"] = data.money_day_fact


    # выкидываем закодированные средними признаки 
    data.drop(["d", "money_day_fact", 'week','money_mean_week'], axis=1, inplace=True)

    data = data[2:]
    data = data.reset_index(drop=True)

    # считаем индекс в датафрейме, после которого начинается тестовыый отрезок
    test_index = int(len(data)-test_size)
    
    # разбиваем весь датасет на тренировочную и тестовую выборку
    X_train = data.loc[:test_index].drop(["y"], axis=1)
    y_train = data.loc[:test_index]["y"]
    X_test = data.loc[test_index:].drop(["y"], axis=1)
    y_test = data.loc[test_index:]["y"]

    return X_train, X_test, y_train, y_test

# COMMAND ----------

X_train, X_test, y_train, y_test = prepareData(df3[:-28], test_size=28)

# COMMAND ----------

y_test.tail()

# COMMAND ----------

# MAGIC %%time
# MAGIC LGBMTimeSeriesCV(X_train, y_train, 5, lgbm, wape, kwargs)

# COMMAND ----------

y_pred1 = lgbm.predict(X_test)
oof = lgbm.predict(X_train)

# COMMAND ----------



# COMMAND ----------

wape(y_pred1, y_test)

# COMMAND ----------

plt.figure(figsize=(25, 10))
plt.plot(df3.d, df3.money_day_fact)
plt.plot(df3.d, df3.Dayforecast)
plt.plot(df3.d, df3.money_mean_week)
plt.plot(df3.d, df3.Weekforecast)
plt.plot(df3.d[-56:-28], y_pred1, label='lgb')
plt.plot(df3.d[2:-55], oof, label='y_hat')
plt.ylim(0,25000)
plt.xlim(1550,1741)
plt.legend(loc="best", fontsize=13)

# COMMAND ----------

grid_df, features_columns = get_data_by_store(STORES_IDS[0])
grid_df.head()

# COMMAND ----------



# COMMAND ----------

df5 = grid_df.groupby('d')['money'].sum().reset_index()
df5.head()

# COMMAND ----------

def get_data_by_store_dept(store, dept):
    
    # Read and contact basic feature
    df = pd.concat([pd.read_pickle(BASE),
                    pd.read_pickle(PRICE).iloc[:,2:],
                    pd.read_pickle(CALENDAR).iloc[:,2:]],
                    axis=1)
    
    # Leave only relevant store
    df = df[(df['store_id']==store)&(df['dept_id']==dept)]
 
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

# COMMAND ----------

#DEPTS ids
DEPTS_IDS = pd.read_csv(ORIGINAL+'sales_train_validation.csv')['dept_id']
DEPTS_IDS = list(DEPTS_IDS.unique())

# COMMAND ----------

STORES_IDS

# COMMAND ----------

DEPTS_IDS

# COMMAND ----------

grid_df.item_id.unique()

# COMMAND ----------

grid_df, features_columns = get_data_by_store_dept(STORES_IDS[0], DEPTS_IDS[-1])
grid_df.head()

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

grid_df['sell_price'] = grid_df['sell_price'].astype(np.float64).round(2)
grid_df['money'] = grid_df.sales * grid_df.sell_price
df3 = grid_df.groupby('d')['money'].sum().reset_index()
data = df3.money[:-84]
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
model = HoltWinters(data[:-28], slen = 7, alpha = alpha_final, beta = beta_final, gamma = gamma_final, n_preds = 56, scaling_factor = 3) #n_preds=28 для графика
model.triple_exponential_smoothing()
print(wape(model.result[-56:-28],data[-28:]).round(2))
df3['Dayforecast'] = np.nan  
df3.iloc[:,-1] = model.result

#     тренд по неделям записываем в дф3
df3['week'] = (df3.d-3)//7
df4 = df3.groupby('week')['money'].sum().reset_index()
data = df4.money[1:-12]
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
model = HoltWinters(data[:-4], slen = 52, alpha = alpha_final, beta = beta_final, gamma = gamma_final, n_preds = 8, scaling_factor = 3) #n_preds=4 для графика
model.triple_exponential_smoothing()
print(wape(model.result[-8:-4],data[-4:]).round(2))
df4['Weekforecast'] = np.nan  
df4.iloc[1:,-1] = model.result
df4.iloc[0,-1] = df4.iloc[1,-1]
df3 = df3.merge(df4, how='left', on=['week'])
df3.columns = ['d', 'money_day_fact', 'Dayforecast', 'week', 'money_mean_week', 'Weekforecast']

# COMMAND ----------

plt.figure(figsize=(25, 10))
plt.plot(df3.d, df3.money_day_fact)
plt.plot(df3.d, df3.Dayforecast)
plt.plot(df3.d, df3.money_mean_week)
plt.plot(df3.d, df3.Weekforecast)
# plt.plot(df3.d[-56:-28], y_pred1, label='lgb')
# plt.plot(df3.d[2:-55], oof, label='y_hat')
# plt.ylim(0,1200)
plt.xlim(1500,1941)
plt.legend(loc="best", fontsize=13)

# COMMAND ----------

