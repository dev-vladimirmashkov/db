# Databricks notebook source
# General imports
import numpy as np
import pandas as pd
import os, sys, gc, time, warnings, pickle, psutil, random

from math import ceil

from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

pd.__version__

# COMMAND ----------

## Simple "Memory profilers" to see memory usage
def get_memory_usage():
    return np.round(psutil.Process(os.getpid()).memory_info()[0]/2.**30, 2) 
        
def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)

# COMMAND ----------

## Memory Reducer
# :df pandas dataframe to reduce size             # type: pd.DataFrame()
# :verbose                                        # type: bool
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                       df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

# COMMAND ----------

## Merging by concat to not lose dtypes
def merge_by_concat(df1, df2, merge_on):
    merged_gf = df1[merge_on]
    merged_gf = merged_gf.merge(df2, on=merge_on, how='left')
    new_columns = [col for col in list(merged_gf) if col not in merge_on]
    df1 = pd.concat([df1, merged_gf[new_columns]], axis=1)
    return df1

# COMMAND ----------

########################### Vars
#################################################################################
TARGET = 'sales'         # Our main target
END_TRAIN = 1913         # Last day in train set
MAIN_INDEX = ['id','d']  # We can identify item by these columns

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
# MAGIC ./azcopy copy "https://bricksresult.blob.core.windows.net/data/20200308grid1941/grid_part_1941_1.pkl?st=2020-03-04T06%3A58%3A48Z&se=2020-12-05T06%3A58%3A00Z&sp=racwdl&sv=2018-03-28&sr=c&sig=LFmu8D%2FVsGZfEj%2F45W6nldF2lvq2mQr9kFl6AblqUN8%3D" "data/grid_part_1941_1.pkl" 
# MAGIC ./azcopy copy "https://bricksresult.blob.core.windows.net/data/20200308wall/calendar.csv?st=2020-03-04T06%3A58%3A48Z&se=2020-12-05T06%3A58%3A00Z&sp=racwdl&sv=2018-03-28&sr=c&sig=LFmu8D%2FVsGZfEj%2F45W6nldF2lvq2mQr9kFl6AblqUN8%3D" "data/calendar.csv" 
# MAGIC ./azcopy copy "https://bricksresult.blob.core.windows.net/data/20200308wall/datasets_619729_1195504_us_disasters_m5.csv?st=2020-03-04T06%3A58%3A48Z&se=2020-12-05T06%3A58%3A00Z&sp=racwdl&sv=2018-03-28&sr=c&sig=LFmu8D%2FVsGZfEj%2F45W6nldF2lvq2mQr9kFl6AblqUN8%3D" "data/datasets_619729_1195504_us_disasters_m5.csv"

# COMMAND ----------

grid_df = pd.read_pickle('data/grid_part_1941_1.pkl')
calendar_df = pd.read_csv('data/calendar.csv')
grid_df.head()

# COMMAND ----------

icols = ['date',
         'd'
         ]
calendar_df['d'] = calendar_df['d'].apply(lambda x: x[2:]).astype(np.int16)
grid_df = grid_df.merge(calendar_df[icols], on=['d'], how='left')
grid_df['date'] = pd.to_datetime(grid_df['date'])

# COMMAND ----------

disasters = pd.read_csv('data/datasets_619729_1195504_us_disasters_m5.csv')

disasters.head()

# COMMAND ----------

disasters['date1'] = pd.to_datetime(disasters.incident_begin_date)
disasters['date2'] = pd.to_datetime(disasters.incident_end_date)

# COMMAND ----------

disasters['date1'] = pd.to_datetime(disasters['date1'].dt.strftime('%Y-%m-%d'))
disasters['date2'] = pd.to_datetime(disasters['date2'].dt.strftime('%Y-%m-%d'))

# COMMAND ----------

disasters.loc[disasters['date2'].isna(), 'date2'] = disasters.loc[disasters['date2'].isna(), 'date1']

# COMMAND ----------



# COMMAND ----------

disasters['state_id'] = disasters['state']

# COMMAND ----------

grid_df.head()

# COMMAND ----------

disasters.head(10)

# COMMAND ----------

disasters.sample(10)

# COMMAND ----------

disasters.incident_type.unique()

# COMMAND ----------

disasters.pa_program_declared.sum()

# COMMAND ----------

len(disasters)

# COMMAND ----------

disasters2 = disasters.groupby(['disaster_number', 'state_id', 'incident_type', 'date1']).agg({'date2':['first'], 'designated_area':['count'], 'pa_program_declared':['sum'], 
                                                                                               'hm_program_declared':['sum'], 'ih_program_declared':['sum']}).reset_index()
disasters2.head()

# COMMAND ----------

disasters2.columns = ['disaster_number', 'state_id', 'incident_type', 'date1','date2','county','pa_program_declared', 'hm_program_declared', 'ih_program_declared']
disasters2.tail()

# COMMAND ----------

disasters3 = disasters2[(disasters2.state_id == 'TX')]
disasters3.head()

# COMMAND ----------

disasters3 = disasters3[~(disasters2.incident_type == 'Other')]
disasters3.head()

# COMMAND ----------

disasters3

# COMMAND ----------

disasters3.index

# COMMAND ----------

disasters2 = disasters3.reset_index(drop=True)
disasters2.columns = ['disaster_number', 'state_id', 'incident_type', 'date1','date2','counties','pa_program_declared', 'hm_program_declared', 'ih_program_declared']
disasters2

# COMMAND ----------

from datetime import timedelta
daterange = pd.DataFrame(pd.date_range(start='2011-01-01', end='2017-01-02').to_series(), columns=['date']).reset_index(drop=True)
daterange['disaster'] = 0
daterange['state_id'] = 'TX'
daterange['incident_type'] = 0
daterange['start_disaster'] = 0
daterange['before_disaster'] = 0
daterange['ih_program_declared'] = 0

for i in disasters2.index:
  daterange.loc[(daterange.date >= disasters2.date1[i])&(daterange.date <= disasters2.date2[i]), 'disaster'] = disasters2.counties[i]
  daterange.loc[(daterange.date >= disasters2.date1[i])&(daterange.date <= disasters2.date2[i]), 'incident_type'] = disasters2.incident_type[i]
  
  daterange.loc[(daterange.date >= disasters2.date1[i])&(daterange.date <= disasters2.date2[i]), 'ih_program_declared'] = disasters2.ih_program_declared[i]
  daterange.loc[(daterange.date == disasters2.date1[i]), 'start_disaster'] = disasters2.counties[i]
  
  
  for l in range(8):
    daterange.loc[(daterange.date == (disasters2.date1[i] - timedelta(days=l))), 'before_disaster'] = l

daterange.head()

# COMMAND ----------

i=1
daterange[(daterange.date >= (disasters2.date1[i]- timedelta(days=l)))&(daterange.date <= disasters2.date2[i])]

# COMMAND ----------

# daterange['incident_type'] = daterange['incident_type'].astype('category')

# COMMAND ----------

grid_df = grid_df.merge(daterange, on=['state_id','date'], how='left')
grid_df.head()

# COMMAND ----------

grid_df.disaster = grid_df.disaster.fillna(0)
grid_df.start_disaster = grid_df.start_disaster.fillna(0)
grid_df.before_disaster = grid_df.before_disaster.fillna(0)
grid_df.ih_program_declared = grid_df.ih_program_declared.fillna(0)

# COMMAND ----------

grid_df.incident_type = grid_df.incident_type.fillna(0)

# COMMAND ----------

grid_df.sample(14)

# COMMAND ----------

daterange.columns

# COMMAND ----------

grid_df['incident_type'] = grid_df['incident_type'].astype('category')

# COMMAND ----------

grid_df1 = grid_df[['id', 'd', 'disaster', 'incident_type', 'start_disaster',
       'before_disaster', 'ih_program_declared']]

grid_df1.to_pickle('grid_part_1941_17_disaster2.pkl')
grid_df1.sample(17)

# COMMAND ----------

# MAGIC %sh
# MAGIC   
# MAGIC ./azcopy copy  "grid_part_1941_17_disaster2.pkl"  "https://bricksresult.blob.core.windows.net/data/20200308grid1941/grid_part_1941_17_disaster2.pkl?st=2020-03-04T06%3A58%3A48Z&se=2020-12-05T06%3A58%3A00Z&sp=racwdl&sv=2018-03-28&sr=c&sig=LFmu8D%2FVsGZfEj%2F45W6nldF2lvq2mQr9kFl6AblqUN8%3D"

# COMMAND ----------

grid_df1.disaster.sum()

# COMMAND ----------

# 76381592.0

# COMMAND ----------

grid_df1.start_disaster.sum()

# COMMAND ----------

# 2586670.0