# Databricks notebook source
pip install fbprophet

# COMMAND ----------

# General imports
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
# warnings.filterwarnings('ignore')
import lightgbm as lgb
from scipy import stats
#from fbprophet import Prophet
import itertools
from sklearn.model_selection import GridSearchCV
#from fbprophet.diagnostics import cross_validation
pd.__version__

# COMMAND ----------

# MAGIC %sh
# MAGIC wget -O azcopy.tar.gz https://aka.ms/downloadazcopy-v10-linux
# MAGIC tar -xf azcopy.tar.gz
# MAGIC cp "$(dirname "$(find . -path ./azcopy_linux\* -type f| tail -1)")"/azcopy azcopy
# MAGIC mkdir data
# MAGIC mkdir model
# MAGIC ./azcopy copy "https://bricksresult.blob.core.windows.net/data/SalesLetoil3/*?sv=2019-12-12&ss=bfqt&srt=sco&sp=rwdlacupx&se=2028-01-11T21:35:49Z&st=2021-01-12T13:35:49Z&spr=https&sig=%2F9iDTc%2FE8MOMfEAvfTYkTpil%2BR5fLzdLkBJyDbJsfWw%3D" "data/"  

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

# COMMAND ----------

# функция WAPE
def wape(y_pred, y_true):
    res = np.sum(np.abs(y_true - y_pred)) / np.abs(np.sum(y_true)) * 100
    return res

# COMMAND ----------

df = read_parquet_folder_as_pandas('data')
df.drop(['marja','sales_model','alcohol','place_Zone','ads','sexid'],1,inplace=True)
df.drop(df[df.salesQ==0].index,inplace=True)
df.head()

# COMMAND ----------

# функция WAPE
def wape(y_pred, y_true):
    res = np.sum(np.abs(y_true - y_pred)) / np.abs(np.sum(y_true)) * 100
    return res

#преобразую год и месяц в один таймстемп 
df['timestamp'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str),
                                              format='%Y-%m')

df = df.sort_values(by='timestamp')

plt.title('Timestamp and SalesQ')
plt.plot(df.timestamp,df.salesQ)

# COMMAND ----------

# todo спрогнозироать продажи по каждому ITEMID на сентябрь - ноябрь 2019 года. Сравнить факт. Посчитать WAPE.

# COMMAND ----------

df['price'] =  df['salesM'] / df['salesQ']
df['price_lag'] = df.price.apply(lambda x: x.shift(1))


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

df.groupby(['ITEMID'])['salesQ'].transform(lambda x: x.shift(3).rolling(6).std())

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

# COMMAND ----------

TARGET = 'salesQ'

features_old = ['SalesQ_lag1', 'SalesM_lag4', 'rolling_mean_M1_3', 'rolling_std_M2_9', 'price', 'rolling_mean_M2_3', 'rolling_std_M1_9', 'rolling_mean_M1_2', 'SalesM_lag2', 'rolling_std_M1_6', 'rolling_std_M3_6', 'rolling_mean_M2_9', 'rolling_std_M2_2', 'rolling_std_M1_2', 'rolling_std_M3_2', 'rolling_mean_M1_6', 'rolling_std_M3_3', 'rolling_mean_M3_6', 'rolling_mean_M2_6']

# COMMAND ----------

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

df.isnull().sum()

# COMMAND ----------

rest_cols = features_
WAPEmean = np.zeros(len(rest_cols))
NumIter = np.zeros(len(rest_cols))
q = 0
used_cols = []
while len(rest_cols) > 0:
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
    if len(used_cols)>20:
      break
    
    rest_cols = [x for x in rest_cols if x != best_col]
    q += 1        

    

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



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# еще не запускал полностью, жду проверк ибо примерное время выпонление этой ячейки 5-6часов
rest_cols = features_
WAPEmean = np.zeros(len(rest_cols))
NumIter = np.zeros(len(rest_cols))
q = 0
used_cols = []
while len(rest_cols) > 0:
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
    if len(used_cols)>20:
      break
    
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

# PROPHET

# COMMAND ----------

df['y_m'] = df.year.astype(str)+df.month.astype(str)
prop_df = df.groupby('timestamp').agg({'salesQ':'sum'})
prop_df = prop_df.sort_index()
prop_df = prop_df.reset_index()
prop_df.columns=['ds','y']
prop_df['y'] = prop_df['y'].astype(float)

print( 'p-val for diki-fuller test. Time Series is stationar'   
      ,sm.tsa.stattools.adfuller(prop_df.set_index('ds').y)[1])

prop_df.head()

# COMMAND ----------

# truncate
trains = prop_df[18:50]
trains.reset_index(inplace=True)
trains.drop('index',axis=1,inplace=True)

tests = prop_df[18:]
tests.reset_index(inplace=True)
tests.drop(['index','y'],axis=1,inplace=True)


# COMMAND ----------

def plotly_df(df,title=''):
  common_kw = dict(x=prop_df.index, mode = 'lines')
  data = [go.Scatter(y=df[c],name=c,**common_kw) for c in df.columns]
  layout = dict(title=title)
  fig = dict(data=data,layout=layout)
  iplot(fig,show_link=False)

plotly_df(prop_df.set_index('ds'),title='sales through yeasrs')

# COMMAND ----------

model = Prophet(weekly_seasonality=False,
         daily_seasonality=False,
               yearly_seasonality=True)
model.fit(trains)

forecast = model.predict(tests)

print('wape is: ', # 126
     wape(forecast.tail(4).yhat.values,
          prop_df.tail(4).y.values))

# COMMAND ----------

model.plot(forecast)

# COMMAND ----------

model.plot_components(forecast)

# COMMAND ----------

def inverse_boxcox(y, lambda_):
    return np.exp(y) if lambda_ == 0 else np.exp(np.log(lambda_ * y + 1) / lambda_)

trains2 = trains.set_index('ds')

#box-cox
trains2['y'], lambda_prophet = stats.boxcox(trains2['y'])
trains2.reset_index(inplace=True)

model2 = Prophet(seasonality_mode='multiplicative')
model2.fit(trains2)
forecast2 = model2.predict(future)

print('wape with y hat and box-cox: ', #99
     wape(forecast2.tail(4).yhat.values,
          prop_df.tail(4).y.values))

# COMMAND ----------

# halt_winters
print('hl')

# COMMAND ----------

hw_df = prop_df[18:].set_index('ds')
hw_df.index= pd.DatetimeIndex(hw_df.index).to_period('M')
hw_train = hw_df[:'2019-08']
hw_train.tail()

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
            

from sklearn.model_selection import TimeSeriesSplit # you have everything done for you

def timeseriesCVscore(params, series, loss_function=mean_squared_error, slen=24):
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
    tscv = TimeSeriesSplit(n_splits=3) 
    
    # iterating over folds, train model on each, forecast and calculate error
    for train, test in tscv.split(values):

        model = HoltWinters(series=values[train], slen=slen, 
                            alpha=alpha, beta=beta, gamma=gamma, n_preds=len(test))
        model.triple_exponential_smoothing()
        
        predictions = model.result[-len(test):]
        actual = values[test]
        error = loss_function(predictions, actual)
        errors.append(error)
        
    return np.mean(np.array(errors))

# COMMAND ----------

from statsmodels.tsa.holtwinters import ExponentialSmoothing

model_hw = ExponentialSmoothing(hw_train, trend='add', seasonal='add', seasonal_periods=12, damped=True).fit(optimized=True ,use_boxcox=True, remove_bias=False)
pred = model_hw.predict(start = hw_df.index[-4],end = hw_df.index[-1])

print('wape: ',wape(pred.values,prop_df.tail(4).y.values))

hw_train.plot()
pred.plot()

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

wape(pred.values,prop_df.tail(4).y.values)

# COMMAND ----------

# initializing model parameters alpha, beta and gamma
x = [0, 0, 0] 

# Minimizing the loss function 
opt = minimize(timeseriesCVscore, x0=x, 
               args=(data, wape), 
               method="TNC", bounds = ((0, 1), (0, 1), (0, 1))
              )

# Take optimal values...
alpha_final, beta_final, gamma_final = opt.x
print(alpha_final, beta_final, gamma_final)

# ...and train the model with them, forecasting for the next 50 hours
model = HoltWinters(data, slen = 24, 
                    alpha = alpha_final, 
                    beta = beta_final, 
                    gamma = gamma_final, 
                    n_preds = 50, scaling_factor = 3)
model.triple_exponential_smoothing()

# COMMAND ----------

prop_df.tail(4).y.values

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# сохранить результат
result2019.to_parquet('data/res2019v0.parquet', index=False)


# COMMAND ----------

# MAGIC %sh
# MAGIC 
# MAGIC ./azcopy copy data/res2019v0.parquet "https://bricksresult.blob.core.windows.net/result/res2019v0.parquet?sv=2019-12-12&ss=bfqt&srt=sco&sp=rwdlacupx&se=2028-01-11T21:35:49Z&st=2021-01-12T13:35:49Z&spr=https&sig=%2F9iDTc%2FE8MOMfEAvfTYkTpil%2BR5fLzdLkBJyDbJsfWw%3D" 