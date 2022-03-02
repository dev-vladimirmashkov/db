# Databricks notebook source
# General imports
import numpy as np
import pandas as pd
import os, sys, gc, time, warnings, pickle, psutil, random
 
# custom imports
from multiprocessing import Pool        # Multiprocess Runs
from functools import partial
 
# plotlib
import seaborn as sns
import matplotlib.pyplot as plt
 
warnings.filterwarnings('ignore')

from typing import Union
from tqdm import tqdm
pd.__version__

# COMMAND ----------

# MAGIC %sh
# MAGIC wget -O azcopy.tar.gz https://aka.ms/downloadazcopy-v10-linux
# MAGIC tar -xf azcopy.tar.gz
# MAGIC cp "$(dirname "$(find . -path ./azcopy_linux\* -type f| tail -1)")"/azcopy azcopy
# MAGIC mkdir data
# MAGIC mkdir data1
# MAGIC mkdir test
# MAGIC mkdir models
# MAGIC ./azcopy copy "https://bricksresult.blob.core.windows.net/data/20200308wall/sales_train_evaluation.csv?st=2020-03-04T06%3A58%3A48Z&se=2020-12-05T06%3A58%3A00Z&sp=racwdl&sv=2018-03-28&sr=c&sig=LFmu8D%2FVsGZfEj%2F45W6nldF2lvq2mQr9kFl6AblqUN8%3D" "data/sales_train_evaluation.csv"
# MAGIC ./azcopy copy "https://bricksresult.blob.core.windows.net/data/20200308wall/calendar.csv?st=2020-03-04T06%3A58%3A48Z&se=2020-12-05T06%3A58%3A00Z&sp=racwdl&sv=2018-03-28&sr=c&sig=LFmu8D%2FVsGZfEj%2F45W6nldF2lvq2mQr9kFl6AblqUN8%3D" "data/calendar.csv"
# MAGIC ./azcopy copy "https://bricksresult.blob.core.windows.net/data/20200308wall/sell_prices.csv?st=2020-03-04T06%3A58%3A48Z&se=2020-12-05T06%3A58%3A00Z&sp=racwdl&sv=2018-03-28&sr=c&sig=LFmu8D%2FVsGZfEj%2F45W6nldF2lvq2mQr9kFl6AblqUN8%3D" "data/sell_prices.csv"
# MAGIC ./azcopy copy "https://bricksresult.blob.core.windows.net/data/20200308wall/sample_submission.csv?st=2020-03-04T06%3A58%3A48Z&se=2020-12-05T06%3A58%3A00Z&sp=racwdl&sv=2018-03-28&sr=c&sig=LFmu8D%2FVsGZfEj%2F45W6nldF2lvq2mQr9kFl6AblqUN8%3D" "data/sample_submission.csv"

# COMMAND ----------

# MAGIC %sh
# MAGIC ./azcopy copy "https://bricksresult.blob.core.windows.net/data/20200308grid1941/submission_vblend17.csv?st=2020-03-04T06%3A58%3A48Z&se=2020-12-05T06%3A58%3A00Z&sp=racwdl&sv=2018-03-28&sr=c&sig=LFmu8D%2FVsGZfEj%2F45W6nldF2lvq2mQr9kFl6AblqUN8%3D" "data/submission_vblend17.csv"
# MAGIC ./azcopy copy "https://bricksresult.blob.core.windows.net/data/20200308grid1941/submission_v1941_21_1.csv?st=2020-03-04T06%3A58%3A48Z&se=2020-12-05T06%3A58%3A00Z&sp=racwdl&sv=2018-03-28&sr=c&sig=LFmu8D%2FVsGZfEj%2F45W6nldF2lvq2mQr9kFl6AblqUN8%3D" "data/submission_v1941_21_1.csv"
# MAGIC ./azcopy copy "https://bricksresult.blob.core.windows.net/data/20200308grid1941/submission_v1941_21_0.csv?st=2020-03-04T06%3A58%3A48Z&se=2020-12-05T06%3A58%3A00Z&sp=racwdl&sv=2018-03-28&sr=c&sig=LFmu8D%2FVsGZfEj%2F45W6nldF2lvq2mQr9kFl6AblqUN8%3D" "data/submission_v1941_21_0.csv"

# COMMAND ----------

# MAGIC %sh
# MAGIC ls data

# COMMAND ----------

df_train_full = pd.read_csv("data/sales_train_evaluation.csv")
df_train_full.iloc[:, -31:].head()

# COMMAND ----------

## evaluation metric
## from https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/133834 and edited to get scores at all levels
class WRMSSEEvaluator(object):

    def __init__(self, train_df: pd.DataFrame, valid_df: pd.DataFrame, calendar: pd.DataFrame, prices: pd.DataFrame):
        train_y = train_df.loc[:, train_df.columns.str.startswith('d_')]
        train_target_columns = train_y.columns.tolist()
        weight_columns = train_y.iloc[:, -28:].columns.tolist()

        train_df['all_id'] = 0  # for lv1 aggregation

        id_columns = train_df.loc[:, ~train_df.columns.str.startswith('d_')].columns.tolist()
        valid_target_columns = valid_df.loc[:, valid_df.columns.str.startswith('d_')].columns.tolist()

        if not all([c in valid_df.columns for c in id_columns]):
            valid_df = pd.concat([train_df[id_columns], valid_df], axis=1, sort=False)

        self.train_df = train_df
        self.valid_df = valid_df
        self.calendar = calendar
        self.prices = prices

        self.weight_columns = weight_columns
        self.id_columns = id_columns
        self.valid_target_columns = valid_target_columns

        weight_df = self.get_weight_df()

        self.group_ids = (
            'all_id',
            'cat_id',
            'state_id',
            'dept_id',
            'store_id',
            'item_id',
            ['state_id', 'cat_id'],
            ['state_id', 'dept_id'],
            ['store_id', 'cat_id'],
            ['store_id', 'dept_id'],
            ['item_id', 'state_id'],
            ['item_id', 'store_id']
        )

        for i, group_id in enumerate(tqdm(self.group_ids)):
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

    def rmsse(self, valid_preds: pd.DataFrame, lv: int) -> pd.Series:
        valid_y = getattr(self, f'lv{lv}_valid_df')
        score = ((valid_y - valid_preds) ** 2).mean(axis=1)
        scale = getattr(self, f'lv{lv}_scale')
        return (score / scale).map(np.sqrt)

    def score(self, valid_preds: Union[pd.DataFrame, np.ndarray]):
        assert self.valid_df[self.valid_target_columns].shape == valid_preds.shape

        if isinstance(valid_preds, np.ndarray):
            valid_preds = pd.DataFrame(valid_preds, columns=self.valid_target_columns)

        valid_preds = pd.concat([self.valid_df[self.id_columns], valid_preds], axis=1, sort=False)

        group_ids = []
        all_scores = []
        for i, group_id in enumerate(self.group_ids):
            lv_scores = self.rmsse(valid_preds.groupby(group_id)[self.valid_target_columns].sum(), i + 1)
            weight = getattr(self, f'lv{i + 1}_weight')
            lv_scores = pd.concat([weight, lv_scores], axis=1, sort=False).prod(axis=1)
            group_ids.append(group_id)
            all_scores.append(lv_scores.sum())

        return group_ids, all_scores

# COMMAND ----------

# MAGIC %sh
# MAGIC ./azcopy copy "https://bricksresult.blob.core.windows.net/data/20200308grid1941/datasets_687702_1205782_m5-forecasting-accuracy-publicleaderboard-rank.csv?st=2020-03-04T06%3A58%3A48Z&se=2020-12-05T06%3A58%3A00Z&sp=racwdl&sv=2018-03-28&sr=c&sig=LFmu8D%2FVsGZfEj%2F45W6nldF2lvq2mQr9kFl6AblqUN8%3D" "data/datasets_687702_1205782_m5-forecasting-accuracy-publicleaderboard-rank.csv"

# COMMAND ----------

## public LB rank
def get_lb_rank(score):
    """
    Get rank on public LB as of 2020-05-31 23:59:59
    """
    df_lb = pd.read_csv("data/datasets_687702_1205782_m5-forecasting-accuracy-publicleaderboard-rank.csv")

    return (df_lb.Score <= score).sum() + 1

# COMMAND ----------

## reading data
df_calendar = pd.read_csv("data/calendar.csv")
df_prices = pd.read_csv("data/sell_prices.csv")
df_sample_submission = pd.read_csv("data/sample_submission.csv")
df_sample_submission["order"] = range(df_sample_submission.shape[0])
df_train = df_train_full.iloc[:, :-28]
df_valid = df_train_full.iloc[:, -28:]

evaluator = WRMSSEEvaluator(df_train, df_valid, df_calendar, df_prices)

df_lb = pd.read_csv("data/datasets_687702_1205782_m5-forecasting-accuracy-publicleaderboard-rank.csv")

# COMMAND ----------

## structure of validation data
preds_valid = df_valid.copy() + np.random.randint(100, size = df_valid.shape)
preds_valid.head()

# COMMAND ----------

def valid(sub):
    preds_valid = sub
    preds_valid = preds_valid[preds_valid.id.str.contains("validation")]
    preds_valid = preds_valid.merge(df_sample_submission[["id", "order"]], on = "id").sort_values("order").drop(["id", "order"], axis = 1).reset_index(drop = True)
    preds_valid.rename(columns = {
        "F1": "d_1914", "F2": "d_1915", "F3": "d_1916", "F4": "d_1917", "F5": "d_1918", "F6": "d_1919", "F7": "d_1920",
        "F8": "d_1921", "F9": "d_1922", "F10": "d_1923", "F11": "d_1924", "F12": "d_1925", "F13": "d_1926", "F14": "d_1927",
        "F15": "d_1928", "F16": "d_1929", "F17": "d_1930", "F18": "d_1931", "F19": "d_1932", "F20": "d_1933", "F21": "d_1934",
        "F22": "d_1935", "F23": "d_1936", "F24": "d_1937", "F25": "d_1938", "F26": "d_1939", "F27": "d_1940", "F28": "d_1941"
    }, inplace = True)
    groups, scores = evaluator.score(preds_valid)
    score_public_lb = np.mean(scores)

    return score_public_lb

# COMMAND ----------

def OOF_StackW(OOF1, OOF2, range1, res_a):
    for i,alpha in enumerate(range1):
        rez1 = OOF1.iloc[:,1:].values
        rez2 = OOF2.iloc[:,1:].values
        oof = rez1*alpha + rez2*(1-alpha)
        sub = OOF1.copy()
        sub.iloc[:,1:] = oof
        res_a[i] = valid(sub)    
    i_min = np.argmin(res_a) 
    return i_min

# COMMAND ----------



# COMMAND ----------

def StackRez(OOF1, OOF2, alpha):    
    rez1 = OOF1.iloc[:,1:].values
    rez2 = OOF2.iloc[:,1:].values
    oof = rez1*alpha + rez2*(1-alpha)
    sub = OOF1.copy()
    sub.iloc[:,1:] = oof
    return sub

# COMMAND ----------

SUBN = pd.read_csv('data/submission_vblend17.csv')
sub210 = pd.read_csv('data/submission_v1941_21_0.csv')
sub211 = pd.read_csv('data/submission_v1941_21_1.csv')
# sub220 = pd.read_csv('data/submission_v1941_22_0.csv')

# COMMAND ----------

valid(sub210)

# COMMAND ----------

valid(sub211)

# COMMAND ----------

# valid(sub220)

# COMMAND ----------

VER = 'blend18'
range1 = np.arange(0,1.8,0.1)
res_a = np.zeros(len(range1))
OOF1 = SUBN
OOF2 = sub210
i_min = OOF_StackW(OOF1, OOF2, range1, res_a)  
alpha_min = range1[i_min]
range1 = np.arange(alpha_min-.1,alpha_min+.1,0.01)
res_a = np.zeros(len(range1))        

i_min = OOF_StackW(OOF1, OOF2, range1, res_a)
alpha_min = range1[i_min]
range1 = np.arange(alpha_min-.01,alpha_min+.01,0.001)
res_a = np.zeros(len(range1))
i_min = OOF_StackW(OOF1, OOF2, range1, res_a)
alpha_min = range1[i_min]
WRMSE_min = res_a[i_min]
SUBN = StackRez(OOF1, OOF2, alpha_min)

print('sub210')
print('alpha_min', alpha_min)    
print('WRMSE_min', WRMSE_min)
print('___________________')

# COMMAND ----------

VER = 'blend18'
range1 = np.arange(0,1.8,0.1)
res_a = np.zeros(len(range1))
OOF1 = SUBN
OOF2 = sub211
i_min = OOF_StackW(OOF1, OOF2, range1, res_a)  
alpha_min = range1[i_min]
range1 = np.arange(alpha_min-.1,alpha_min+.1,0.01)
res_a = np.zeros(len(range1))        

i_min = OOF_StackW(OOF1, OOF2, range1, res_a)
alpha_min = range1[i_min]
range1 = np.arange(alpha_min-.01,alpha_min+.01,0.001)
res_a = np.zeros(len(range1))
i_min = OOF_StackW(OOF1, OOF2, range1, res_a)
alpha_min = range1[i_min]
WRMSE_min = res_a[i_min]
SUBN = StackRez(OOF1, OOF2, alpha_min)

print('sub210')
print('alpha_min', alpha_min)    
print('WRMSE_min', WRMSE_min)
print('___________________')

# COMMAND ----------

SUBN.tail(14)

# COMMAND ----------

# MAGIC %sh
# MAGIC pip install bokeh

# COMMAND ----------

## importing packages
import numpy as np
import pandas as pd

from bokeh.layouts import column, row
from bokeh.models import LinearAxis, Range1d, Span
from bokeh.models.tools import HoverTool
from bokeh.plotting import ColumnDataSource, figure, output_notebook, show

from math import pi
from typing import Union
from tqdm import tqdm

output_notebook()

LB_DATES = list(pd.date_range(start = "2016-04-25", end = "2016-05-22").strftime("%Y-%m-%d"))
LB_WEEKDAYS = pd.to_datetime(LB_DATES).to_series().dt.day_name()

# COMMAND ----------

df_submission = SUBN
df_submission.head()

# COMMAND ----------

## evaluation metric
## edited from https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/133834
class WRMSSEEvaluator(object):

    def __init__(self, train_df: pd.DataFrame, valid_df: pd.DataFrame, calendar: pd.DataFrame, prices: pd.DataFrame):
        train_y = train_df.loc[:, train_df.columns.str.startswith('d_')]
        train_target_columns = train_y.columns.tolist()
        weight_columns = train_y.iloc[:, -28:].columns.tolist()

        train_df['all_id'] = 0  # for lv1 aggregation

        id_columns = train_df.loc[:, ~train_df.columns.str.startswith('d_')].columns.tolist()
        valid_target_columns = valid_df.loc[:, valid_df.columns.str.startswith('d_')].columns.tolist()

        if not all([c in valid_df.columns for c in id_columns]):
            valid_df = pd.concat([train_df[id_columns], valid_df], axis=1, sort=False)

        self.train_df = train_df
        self.valid_df = valid_df
        self.calendar = calendar
        self.prices = prices

        self.weight_columns = weight_columns
        self.id_columns = id_columns
        self.valid_target_columns = valid_target_columns

        weight_df = self.get_weight_df()

        self.group_ids = (
            'all_id',
            'cat_id',
            'state_id',
            'dept_id',
            'store_id',
            'item_id',
            ['state_id', 'cat_id'],
            ['state_id', 'dept_id'],
            ['store_id', 'cat_id'],
            ['store_id', 'dept_id'],
            ['item_id', 'state_id'],
            ['item_id', 'store_id']
        )

        for i, group_id in enumerate(tqdm(self.group_ids)):
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

    def rmsse(self, valid_preds: pd.DataFrame, lv: int) -> pd.Series:
        valid_y = getattr(self, f'lv{lv}_valid_df')
        score = ((valid_y - valid_preds) ** 2).mean(axis=1)
        scale = getattr(self, f'lv{lv}_scale')
        return (score / scale).map(np.sqrt)

    def score(self, valid_preds: Union[pd.DataFrame, np.ndarray]):
        assert self.valid_df[self.valid_target_columns].shape == valid_preds.shape

        if isinstance(valid_preds, np.ndarray):
            valid_preds = pd.DataFrame(valid_preds, columns=self.valid_target_columns)

        valid_preds = pd.concat([self.valid_df[self.id_columns], valid_preds], axis=1, sort=False)

        group_ids = []
        all_scores = []

        for i, group_id in enumerate(self.group_ids):
            lv_scores = self.rmsse(valid_preds.groupby(group_id)[self.valid_target_columns].sum(), i + 1)
            weight = getattr(self, f'lv{i + 1}_weight')
            lv_scores = pd.concat([weight, lv_scores], axis=1, sort=False).prod(axis=1)
            group_ids.append(group_id)
            all_scores.append(lv_scores.sum())

        return group_ids, all_scores
    
    def get_scores(self, valid_preds: Union[pd.DataFrame, np.ndarray], lv: int):
        assert self.valid_df[self.valid_target_columns].shape == valid_preds.shape

        if isinstance(valid_preds, np.ndarray):
            valid_preds = pd.DataFrame(valid_preds, columns=self.valid_target_columns)

        valid_preds = pd.concat([self.valid_df[self.id_columns], valid_preds], axis=1, sort=False)
        
        for i, group_id in enumerate(self.group_ids):
            if lv == i+1:
                valid_df = valid_preds.groupby(group_id)[self.valid_target_columns].sum()
                valid_y = getattr(self, f"lv{lv}_valid_df")
                scale = getattr(self, f"lv{lv}_scale")
                weight = getattr(self, f"lv{lv}_weight")
                valid_df["score"] = (((valid_y - valid_df) ** 2).mean(axis = 1) / scale).map(np.sqrt)
                valid_df.columns = ["pred_d_" + str(x) for x in range(1914, 1942)] + ["score"]
                valid_df = pd.concat([valid_df, valid_y], axis = 1)
                valid_df["score_weighted"] = valid_df.score * weight
                valid_df["score_percentage"] = valid_df.score_weighted / valid_df.score_weighted.sum()

        return valid_df.reset_index()

# COMMAND ----------

df_sample_submission["order"] = range(df_sample_submission.shape[0])

df_train = df_train_full.iloc[:, :-28]
df_valid = df_train_full.iloc[:, -28:]

evaluator = WRMSSEEvaluator(df_train, df_valid, df_calendar, df_prices)

# COMMAND ----------

## evaluating submission from public kernel M5 First Public Notebook Under 0.50
## from https://www.kaggle.com/kneroma/m5-first-public-notebook-under-0-50
preds_valid = df_submission[df_submission.id.str.contains("validation")]
preds_valid = preds_valid.merge(df_sample_submission[["id", "order"]], on = "id").sort_values("order").drop(["id", "order"], axis = 1).reset_index(drop = True)
preds_valid.rename(columns = {
    "F1": "d_1914", "F2": "d_1915", "F3": "d_1916", "F4": "d_1917", "F5": "d_1918", "F6": "d_1919", "F7": "d_1920",
    "F8": "d_1921", "F9": "d_1922", "F10": "d_1923", "F11": "d_1924", "F12": "d_1925", "F13": "d_1926", "F14": "d_1927",
    "F15": "d_1928", "F16": "d_1929", "F17": "d_1930", "F18": "d_1931", "F19": "d_1932", "F20": "d_1933", "F21": "d_1934",
    "F22": "d_1935", "F23": "d_1936", "F24": "d_1937", "F25": "d_1938", "F26": "d_1939", "F27": "d_1940", "F28": "d_1941"
}, inplace = True)

groups, scores = evaluator.score(preds_valid)

score_public_lb = np.mean(scores)

for i in range(len(groups)):
    print(f"Score for group {groups[i]}: {round(scores[i], 5)}")

print(f"\nPublic LB Score: {round(score_public_lb, 5)}")

# COMMAND ----------

levels = ["all", "category", "state", "department", "store", "item", "state_category", "state_department",
          "store_category", "store_department", "state_item", "store_item"]

df_levels = pd.DataFrame({
    "level": levels,
    "score": scores
})

source_1 = ColumnDataSource(data = dict(
    level = df_levels.level.values,
    score = df_levels.score.values
))

tooltips = [
    ("Level", "@level"),
    ("Score", "@score")
]

v1 = figure(plot_width = 650, plot_height = 400, y_range = df_levels.level.values, tooltips = tooltips, title = "Scores by all aggregation levels")
v1.hbar(y = "level", right = "score", source = source_1, height = 0.75, alpha = 0.6, legend_label = "Public LB Score")

mean = Span(location = np.mean(df_levels.score.values), dimension = "height", line_color = "grey", line_dash = "dashed", line_width = 1.5)
v1.add_layout(mean)

v1.xaxis.axis_label = "WRMSSE Score"
v1.yaxis.axis_label = "Aggregation Level"

v1.legend.location = "bottom_right"


df_levels.sort_values("score", inplace = True)

source_2 = ColumnDataSource(data = dict(
    level = df_levels.level.values,
    score = df_levels.score.values
))

v2 = figure(plot_width = 650, plot_height = 400, y_range = df_levels.level.values, tooltips = tooltips, title = "Scores by all aggregation levels (sorted)")
v2.hbar(y = "level", right = "score", source = source_2, height = 0.75, alpha = 0.6, legend_label = "Public LB Score")

mean = Span(location = np.mean(df_levels.score.values), dimension = "height", line_color = "grey", line_dash = "dashed", line_width = 1.5)
v2.add_layout(mean)

v2.xaxis.axis_label = "WRMSSE Score"
v2.yaxis.axis_label = "Aggregation Level"

v2.legend.location = "bottom_right"


df_items = df_levels[df_levels.level.str.contains("item")]

source_3 = ColumnDataSource(data = dict(
    level = df_items.level.values,
    score = df_items.score.values
))

v3 = figure(plot_width = 330, plot_height = 200, y_range = df_items.level.values, x_range = Range1d(0, 1), tooltips = tooltips, title = "Scores by item levels")
v3.hbar(y = "level", right = "score", source = source_3, height = 0.75, color = "mediumseagreen", alpha = 0.6)

mean = Span(location = np.mean(df_items.score.values), dimension = "height", line_color = "grey", line_dash = "dashed", line_width = 1.5)
v3.add_layout(mean)

v3.xaxis.axis_label = "WRMSSE Score"
v3.yaxis.axis_label = "Aggregation Level"


df_stores = df_levels[df_levels.level.str.contains("store")]

source_4 = ColumnDataSource(data = dict(
    level = df_stores.level.values,
    score = df_stores.score.values
))

v4 = figure(plot_width = 330, plot_height = 200, y_range = df_stores.level.values, x_range = Range1d(0, 1), tooltips = tooltips, title = "Scores by store levels")
v4.hbar(y = "level", right = "score", source = source_4, height = 0.75, color = "mediumseagreen", alpha = 0.6)

mean = Span(location = np.mean(df_stores.score.values), dimension = "height", line_color = "grey", line_dash = "dashed", line_width = 1.5)
v4.add_layout(mean)

v4.xaxis.axis_label = "WRMSSE Score"
v4.yaxis.axis_label = "Aggregation Level"


df_departments = df_levels[df_levels.level.str.contains("dep")]

source_5 = ColumnDataSource(data = dict(
    level = df_departments.level.values,
    score = df_departments.score.values
))

v5 = figure(plot_width = 330, plot_height = 200, y_range = df_departments.level.values, x_range = Range1d(0, 1), tooltips = tooltips, title = "Scores by department levels")
v5.hbar(y = "level", right = "score", source = source_5, height = 0.75, color = "mediumseagreen", alpha = 0.6)

mean = Span(location = np.mean(df_departments.score.values), dimension = "height", line_color = "grey", line_dash = "dashed", line_width = 1.5)
v5.add_layout(mean)

v5.xaxis.axis_label = "WRMSSE Score"
v5.yaxis.axis_label = "Aggregation Level"


df_states = df_levels[df_levels.level.str.contains("state")]

source_6 = ColumnDataSource(data = dict(
    level = df_states.level.values,
    score = df_states.score.values
))

v6 = figure(plot_width = 330, plot_height = 200, y_range = df_states.level.values, x_range = Range1d(0, 1), tooltips = tooltips, title = "Scores by state levels")
v6.hbar(y = "level", right = "score", source = source_6, height = 0.75, color = "mediumseagreen", alpha = 0.6)

mean = Span(location = np.mean(df_states.score.values), dimension = "height", line_color = "grey", line_dash = "dashed", line_width = 1.5)
v6.add_layout(mean)

v6.xaxis.axis_label = "WRMSSE Score"
v6.yaxis.axis_label = "Aggregation Level"


df_categories = df_levels[df_levels.level.str.contains("cat")]

source_7 = ColumnDataSource(data = dict(
    level = df_categories.level.values,
    score = df_categories.score.values
))

v7 = figure(plot_width = 330, plot_height = 200, y_range = df_categories.level.values, x_range = Range1d(0, 1), tooltips = tooltips, title = "Scores by category levels")
v7.hbar(y = "level", right = "score", source = source_7, height = 0.75, color = "mediumseagreen", alpha = 0.6)

mean = Span(location = np.mean(df_categories.score.values), dimension = "height", line_color = "grey", line_dash = "dashed", line_width = 1.5)
v7.add_layout(mean)

v7.xaxis.axis_label = "WRMSSE Score"
v7.yaxis.axis_label = "Aggregation Level"


show(column(v1, v2, row(v3, v4), row(v5, v6), v7))

# COMMAND ----------

show(column(v1)) 
file_html()
displayHTML()

# COMMAND ----------

from bokeh.plotting import figure, output_file
from bokeh.embed import components, file_html
from bokeh.resources import CDN

# COMMAND ----------

output_file("all.html")
show(column(v1, v2, row(v3, v4), row(v5, v6), v7))
with open("all.html",'r') as f:
    html = f.read()
displayHTML(html)

# COMMAND ----------

df_store_item = evaluator.get_scores(preds_valid, 12)
df_store_item["store_item_id"] = df_store_item.store_id + "-" + df_store_item.item_id

df_store_item_best = df_store_item.sort_values("score_weighted").head(10).rename(columns = {"score_weighted": "score_best"})
df_store_item_worst = df_store_item.sort_values("score_weighted", ascending = False).head(10).sort_values("score_weighted").rename(columns = {"score_weighted": "score_worst"})

df_store_item_best_worst = pd.concat([df_store_item_best, df_store_item_worst])

source_1 = ColumnDataSource(data = dict(
    store_item_id = df_store_item_best_worst.store_item_id.values,
    score_best = df_store_item_best_worst.score_best.values,
    score_worst = df_store_item_best_worst.score_worst.values
))

tooltips_1 = [
    ("Store-Item", "@store_item_id"),
    ("Score", "@score_best{0.0000}")
]

tooltips_2 = [
    ("Store-Item", "@store_item_id"),
    ("Score", "@score_worst{0.0000}")
]

v1 = figure(plot_width = 700, plot_height = 400, y_range = df_store_item_best_worst.store_item_id.values, title = "Best and Worst Store-Item")
v11 = v1.hbar("store_item_id", right = "score_best", source = source_1, height = 0.75, alpha = 0.6, color = "green")
v12 = v1.hbar("store_item_id", right = "score_worst", source = source_1, height = 0.75, alpha = 0.6, color = "red")

v1.add_tools(HoverTool(renderers = [v11], tooltips = tooltips_1))
v1.add_tools(HoverTool(renderers = [v12], tooltips = tooltips_2))

v1.xaxis.axis_label = "WRMSSE Score"
v1.yaxis.axis_label = "Store-Item"


def get_store_item_plot(df, store_item_id):
    """
    Plots the actual and predicted values of store-item
    """
    actual_dates = ["d_" + str(x) for x in range(1914, 1942)]
    predicted_dates = ["pred_d_" + str(x) for x in range(1914, 1942)]
    
    source = ColumnDataSource(data = dict(
        date_number = actual_dates,
        date = LB_DATES,
        weekday = LB_WEEKDAYS,
        actual = df.loc[df.store_item_id == store_item_id, actual_dates].values[0],
        predicted = df.loc[df.store_item_id == store_item_id, predicted_dates].values[0]
    ))
    
    tooltips = [
        ("Date", "@date"),
        ("Weekday", "@weekday"),
        ("Actual", "@actual{0}"),
        ("Predicted", "@predicted{0.0}")
    ]
    
    v = figure(plot_width = 700, plot_height = 400, x_range = actual_dates, tooltips = tooltips, title = f"Store-Item: {store_item_id}")
    v.line("date_number", "actual", source = source, color = "steelblue", alpha = 0.6, width = 3, legend_label = "Actual")
    v.line("date_number", "predicted", source = source, color = "coral", alpha = 0.6, width = 3, legend_label = "Predicted")
    
    v.xaxis.axis_label = "Date"
    v.yaxis.axis_label = "Sales"

    v.xaxis.major_label_orientation = pi / 4
    
    return v

v2 = get_store_item_plot(df_store_item, df_store_item_worst.store_item_id.values[-1])
v3 = get_store_item_plot(df_store_item, df_store_item_worst.store_item_id.values[-2])
v4 = get_store_item_plot(df_store_item, df_store_item_worst.store_item_id.values[-3])
v5 = get_store_item_plot(df_store_item, df_store_item_worst.store_item_id.values[-4])
v6 = get_store_item_plot(df_store_item, df_store_item_worst.store_item_id.values[-5])
v7 = get_store_item_plot(df_store_item, df_store_item_worst.store_item_id.values[-6])
v8 = get_store_item_plot(df_store_item, df_store_item_worst.store_item_id.values[-7])
v9 = get_store_item_plot(df_store_item, df_store_item_worst.store_item_id.values[-8])
v10 = get_store_item_plot(df_store_item, df_store_item_worst.store_item_id.values[-9])
v11 = get_store_item_plot(df_store_item, df_store_item_worst.store_item_id.values[-10])


show(column(v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11))

# COMMAND ----------

output_file("Store_Item.html")
show(column(v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11))
with open("Store_Item.html",'r') as f:
    html = f.read()
displayHTML(html)

# COMMAND ----------

df_state_item = evaluator.get_scores(preds_valid, 11)
df_state_item["state_item_id"] = df_state_item.state_id + "-" + df_state_item.item_id

df_state_item_best = df_state_item.sort_values("score_weighted").head(10).rename(columns = {"score_weighted": "score_best"})
df_state_item_worst = df_state_item.sort_values("score_weighted", ascending = False).head(10).sort_values("score_weighted").rename(columns = {"score_weighted": "score_worst"})

df_state_item_best_worst = pd.concat([df_state_item_best, df_state_item_worst])

source_1 = ColumnDataSource(data = dict(
    state_item_id = df_state_item_best_worst.state_item_id.values,
    score_best = df_state_item_best_worst.score_best.values,
    score_worst = df_state_item_best_worst.score_worst.values
))

tooltips_1 = [
    ("State-Item", "@state_item_id"),
    ("Score", "@score_best{0.0000}")
]

tooltips_2 = [
    ("State-Item", "@state_item_id"),
    ("Score", "@score_worst{0.0000}")
]

v1 = figure(plot_width = 700, plot_height = 400, y_range = df_state_item_best_worst.state_item_id.values, title = "Best and Worst State-Item")
v11 = v1.hbar("state_item_id", right = "score_best", source = source_1, height = 0.75, alpha = 0.6, color = "green")
v12 = v1.hbar("state_item_id", right = "score_worst", source = source_1, height = 0.75, alpha = 0.6, color = "red")

v1.add_tools(HoverTool(renderers = [v11], tooltips = tooltips_1))
v1.add_tools(HoverTool(renderers = [v12], tooltips = tooltips_2))

v1.xaxis.axis_label = "WRMSSE Score"
v1.yaxis.axis_label = "State-Item"


def get_state_item_plot(df, state_item_id):
    """
    Plots the actual and predicted values of state-item
    """
    actual_dates = ["d_" + str(x) for x in range(1914, 1942)]
    predicted_dates = ["pred_d_" + str(x) for x in range(1914, 1942)]
    
    source = ColumnDataSource(data = dict(
        date_number = actual_dates,
        date = LB_DATES,
        weekday = LB_WEEKDAYS,
        actual = df.loc[df.state_item_id == state_item_id, actual_dates].values[0],
        predicted = df.loc[df.state_item_id == state_item_id, predicted_dates].values[0]
    ))
    
    tooltips = [
        ("Date", "@date"),
        ("Weekday", "@weekday"),
        ("Actual", "@actual{0}"),
        ("Predicted", "@predicted{0.0}")
    ]
    
    v = figure(plot_width = 700, plot_height = 400, x_range = actual_dates, tooltips = tooltips, title = f"State-Item: {state_item_id}")
    v.line("date_number", "actual", source = source, color = "steelblue", alpha = 0.6, width = 3, legend_label = "Actual")
    v.line("date_number", "predicted", source = source, color = "coral", alpha = 0.6, width = 3, legend_label = "Predicted")

    v.xaxis.axis_label = "Date"
    v.yaxis.axis_label = "Sales"

    v.xaxis.major_label_orientation = pi / 4

    return v

v2 = get_state_item_plot(df_state_item, df_state_item_worst.state_item_id.values[-1])
v3 = get_state_item_plot(df_state_item, df_state_item_worst.state_item_id.values[-2])
v4 = get_state_item_plot(df_state_item, df_state_item_worst.state_item_id.values[-3])
v5 = get_state_item_plot(df_state_item, df_state_item_worst.state_item_id.values[-4])
v6 = get_state_item_plot(df_state_item, df_state_item_worst.state_item_id.values[-5])
v7 = get_state_item_plot(df_state_item, df_state_item_worst.state_item_id.values[-6])
v8 = get_state_item_plot(df_state_item, df_state_item_worst.state_item_id.values[-7])
v9 = get_state_item_plot(df_state_item, df_state_item_worst.state_item_id.values[-8])
v10 = get_state_item_plot(df_state_item, df_state_item_worst.state_item_id.values[-9])
v11 = get_state_item_plot(df_state_item, df_state_item_worst.state_item_id.values[-10])


show(column(v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11))

# COMMAND ----------

output_file("State_Item.html")
show(column(v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11))
with open("State_Item.html",'r') as f:
    html = f.read()
displayHTML(html)

# COMMAND ----------

SUBN[SUBN.id == 'FOODS_3_702_CA_2_validation']

# COMMAND ----------

df_item = evaluator.get_scores(preds_valid, 6)

df_item_best = df_item.sort_values("score_weighted").head(10).rename(columns = {"score_weighted": "score_best"})
df_item_worst = df_item.sort_values("score_weighted", ascending = False).head(10).sort_values("score_weighted").rename(columns = {"score_weighted": "score_worst"})

df_item_best_worst = pd.concat([df_item_best, df_item_worst])

source_1 = ColumnDataSource(data = dict(
    item_id = df_item_best_worst.item_id.values,
    score_best = df_item_best_worst.score_best.values,
    score_worst = df_item_best_worst.score_worst.values
))

tooltips_1 = [
    ("Item", "@item_id"),
    ("Score", "@score_best{0.0000}")
]

tooltips_2 = [
    ("Item", "@item_id"),
    ("Score", "@score_worst{0.0000}")
]

v1 = figure(plot_width = 700, plot_height = 400, y_range = df_item_best_worst.item_id.values, title = "Best and Worst Item")
v11 = v1.hbar("item_id", right = "score_best", source = source_1, height = 0.75, alpha = 0.6, color = "green")
v12 = v1.hbar("item_id", right = "score_worst", source = source_1, height = 0.75, alpha = 0.6, color = "red")

v1.add_tools(HoverTool(renderers = [v11], tooltips = tooltips_1))
v1.add_tools(HoverTool(renderers = [v12], tooltips = tooltips_2))

v1.xaxis.axis_label = "WRMSSE Score"
v1.yaxis.axis_label = "Item"


def get_item_plot(df, item_id):
    """
    Plots the actual and predicted values of item_id
    """
    actual_dates = ["d_" + str(x) for x in range(1914, 1942)]
    predicted_dates = ["pred_d_" + str(x) for x in range(1914, 1942)]
    
    source = ColumnDataSource(data = dict(
        date_number = actual_dates,
        date = LB_DATES,
        weekday = LB_WEEKDAYS,
        actual = df.loc[df.item_id == item_id, actual_dates].values[0],
        predicted = df.loc[df.item_id == item_id, predicted_dates].values[0]
    ))
    
    tooltips = [
        ("Date", "@date"),
        ("Weekday", "@weekday"),
        ("Actual", "@actual{0}"),
        ("Predicted", "@predicted{0.0}")
    ]
    
    v = figure(plot_width = 700, plot_height = 400, x_range = actual_dates, tooltips = tooltips, title = f"Item: {item_id}")
    v.line("date_number", "actual", source = source, color = "steelblue", alpha = 0.6, width = 3, legend_label = "Actual")
    v.line("date_number", "predicted", source = source, color = "coral", alpha = 0.6, width = 3, legend_label = "Predicted")

    v.xaxis.axis_label = "Date"
    v.yaxis.axis_label = "Sales"

    v.xaxis.major_label_orientation = pi / 4

    return v

v2 = get_item_plot(df_item, df_item_worst.item_id.values[-1])
v3 = get_item_plot(df_item, df_item_worst.item_id.values[-2])
v4 = get_item_plot(df_item, df_item_worst.item_id.values[-3])
v5 = get_item_plot(df_item, df_item_worst.item_id.values[-4])
v6 = get_item_plot(df_item, df_item_worst.item_id.values[-5])
v7 = get_item_plot(df_item, df_item_worst.item_id.values[-6])
v8 = get_item_plot(df_item, df_item_worst.item_id.values[-7])
v9 = get_item_plot(df_item, df_item_worst.item_id.values[-8])
v10 = get_item_plot(df_item, df_item_worst.item_id.values[-9])
v11 = get_item_plot(df_item, df_item_worst.item_id.values[-10])


show(column(v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11))

# COMMAND ----------

output_file("Item.html")
show(column(v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11))
with open("Item.html",'r') as f:
    html = f.read()
displayHTML(html)

# COMMAND ----------

df_store_dept = evaluator.get_scores(preds_valid, 10)
df_store_dept["store_dept_id"] = df_store_dept.store_id + "-" + df_store_dept.dept_id

df_store_dept_best = df_store_dept.sort_values("score_weighted").head(10).rename(columns = {"score_weighted": "score_best"})
df_store_dept_worst = df_store_dept.sort_values("score_weighted", ascending = False).head(10).sort_values("score_weighted").rename(columns = {"score_weighted": "score_worst"})

df_store_dept_best_worst = pd.concat([df_store_dept_best, df_store_dept_worst])

source_1 = ColumnDataSource(data = dict(
    store_dept_id = df_store_dept_best_worst.store_dept_id.values,
    score_best = df_store_dept_best_worst.score_best.values,
    score_worst = df_store_dept_best_worst.score_worst.values
))

tooltips_1 = [
    ("Store-Department", "@store_dept_id"),
    ("Score", "@score_best{0.0000}")
]

tooltips_2 = [
    ("Store-Department", "@store_dept_id"),
    ("Score", "@score_worst{0.0000}")
]

v1 = figure(plot_width = 700, plot_height = 400, y_range = df_store_dept_best_worst.store_dept_id.values, title = "Best and Worst Store-Department")
v11 = v1.hbar("store_dept_id", right = "score_best", source = source_1, height = 0.75, alpha = 0.6, color = "green")
v12 = v1.hbar("store_dept_id", right = "score_worst", source = source_1, height = 0.75, alpha = 0.6, color = "red")

v1.add_tools(HoverTool(renderers = [v11], tooltips = tooltips_1))
v1.add_tools(HoverTool(renderers = [v12], tooltips = tooltips_2))

v1.xaxis.axis_label = "WRMSSE Score"
v1.yaxis.axis_label = "Store-Department"


def get_store_dept_plot(df, store_dept_id):
    """
    Plots the actual and predicted values of store-dept
    """
    actual_dates = ["d_" + str(x) for x in range(1914, 1942)]
    predicted_dates = ["pred_d_" + str(x) for x in range(1914, 1942)]
    
    source = ColumnDataSource(data = dict(
        date_number = actual_dates,
        date = LB_DATES,
        weekday = LB_WEEKDAYS,
        actual = df.loc[df.store_dept_id == store_dept_id, actual_dates].values[0],
        predicted = df.loc[df.store_dept_id == store_dept_id, predicted_dates].values[0]
    ))
    
    tooltips = [
        ("Date", "@date"),
        ("Weekday", "@weekday"),
        ("Actual", "@actual{0}"),
        ("Predicted", "@predicted{0.0}")
    ]
    
    v = figure(plot_width = 700, plot_height = 400, x_range = actual_dates, tooltips = tooltips, title = f"Store-Dept: {store_dept_id}")
    v.line("date_number", "actual", source = source, color = "steelblue", alpha = 0.6, width = 3, legend_label = "Actual")
    v.line("date_number", "predicted", source = source, color = "coral", alpha = 0.6, width = 3, legend_label = "Predicted")

    v.xaxis.axis_label = "Date"
    v.yaxis.axis_label = "Sales"

    v.xaxis.major_label_orientation = pi / 4

    return v

v2 = get_store_dept_plot(df_store_dept, df_store_dept_worst.store_dept_id.values[-1])
v3 = get_store_dept_plot(df_store_dept, df_store_dept_worst.store_dept_id.values[-2])
v4 = get_store_dept_plot(df_store_dept, df_store_dept_worst.store_dept_id.values[-3])
v5 = get_store_dept_plot(df_store_dept, df_store_dept_worst.store_dept_id.values[-4])
v6 = get_store_dept_plot(df_store_dept, df_store_dept_worst.store_dept_id.values[-5])
v7 = get_store_dept_plot(df_store_dept, df_store_dept_worst.store_dept_id.values[-6])
v8 = get_store_dept_plot(df_store_dept, df_store_dept_worst.store_dept_id.values[-7])
v9 = get_store_dept_plot(df_store_dept, df_store_dept_worst.store_dept_id.values[-8])
v10 = get_store_dept_plot(df_store_dept, df_store_dept_worst.store_dept_id.values[-9])
v11 = get_store_dept_plot(df_store_dept, df_store_dept_worst.store_dept_id.values[-10])


show(column(v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11))

# COMMAND ----------

df_store_cat = evaluator.get_scores(preds_valid, 9)
df_store_cat["store_cat_id"] = df_store_cat.store_id + "-" + df_store_cat.cat_id

df_store_cat_best = df_store_cat.sort_values("score_weighted").head(10).rename(columns = {"score_weighted": "score_best"})
df_store_cat_worst = df_store_cat.sort_values("score_weighted", ascending = False).head(10).sort_values("score_weighted").rename(columns = {"score_weighted": "score_worst"})

df_store_cat_best_worst = pd.concat([df_store_cat_best, df_store_cat_worst])

source_1 = ColumnDataSource(data = dict(
    store_cat_id = df_store_cat_best_worst.store_cat_id.values,
    score_best = df_store_cat_best_worst.score_best.values,
    score_worst = df_store_cat_best_worst.score_worst.values
))

tooltips_1 = [
    ("Store-Category", "@store_cat_id"),
    ("Score", "@score_best{0.0000}")
]

tooltips_2 = [
    ("Store-Category", "@store_cat_id"),
    ("Score", "@score_worst{0.0000}")
]

v1 = figure(plot_width = 700, plot_height = 400, y_range = df_store_cat_best_worst.store_cat_id.values, title = "Best and Worst Store-Category")
v11 = v1.hbar("store_cat_id", right = "score_best", source = source_1, height = 0.75, alpha = 0.6, color = "green")
v12 = v1.hbar("store_cat_id", right = "score_worst", source = source_1, height = 0.75, alpha = 0.6, color = "red")

v1.add_tools(HoverTool(renderers = [v11], tooltips = tooltips_1))
v1.add_tools(HoverTool(renderers = [v12], tooltips = tooltips_2))

v1.xaxis.axis_label = "WRMSSE Score"
v1.yaxis.axis_label = "Store-Category"


def get_store_cat_plot(df, store_cat_id):
    """
    Plots the actual and predicted values of store-cat
    """
    actual_dates = ["d_" + str(x) for x in range(1914, 1942)]
    predicted_dates = ["pred_d_" + str(x) for x in range(1914, 1942)]
    
    source = ColumnDataSource(data = dict(
        date_number = actual_dates,
        date = LB_DATES,
        weekday = LB_WEEKDAYS,
        actual = df.loc[df.store_cat_id == store_cat_id, actual_dates].values[0],
        predicted = df.loc[df.store_cat_id == store_cat_id, predicted_dates].values[0]
    ))
    
    tooltips = [
        ("Date", "@date"),
        ("Weekday", "@weekday"),
        ("Actual", "@actual{0}"),
        ("Predicted", "@predicted{0.0}")
    ]
    
    v = figure(plot_width = 700, plot_height = 400, x_range = actual_dates, tooltips = tooltips, title = f"Store-Category: {store_cat_id}")
    v.line("date_number", "actual", source = source, color = "steelblue", alpha = 0.6, width = 3, legend_label = "Actual")
    v.line("date_number", "predicted", source = source, color = "coral", alpha = 0.6, width = 3, legend_label = "Predicted")

    v.xaxis.axis_label = "Date"
    v.yaxis.axis_label = "Sales"

    v.xaxis.major_label_orientation = pi / 4

    return v

v2 = get_store_cat_plot(df_store_cat, df_store_cat_worst.store_cat_id.values[-1])
v3 = get_store_cat_plot(df_store_cat, df_store_cat_worst.store_cat_id.values[-2])
v4 = get_store_cat_plot(df_store_cat, df_store_cat_worst.store_cat_id.values[-3])
v5 = get_store_cat_plot(df_store_cat, df_store_cat_worst.store_cat_id.values[-4])
v6 = get_store_cat_plot(df_store_cat, df_store_cat_worst.store_cat_id.values[-5])
v7 = get_store_cat_plot(df_store_cat, df_store_cat_worst.store_cat_id.values[-6])
v8 = get_store_cat_plot(df_store_cat, df_store_cat_worst.store_cat_id.values[-7])
v9 = get_store_cat_plot(df_store_cat, df_store_cat_worst.store_cat_id.values[-8])
v10 = get_store_cat_plot(df_store_cat, df_store_cat_worst.store_cat_id.values[-9])
v11 = get_store_cat_plot(df_store_cat, df_store_cat_worst.store_cat_id.values[-10])


show(column(v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11))