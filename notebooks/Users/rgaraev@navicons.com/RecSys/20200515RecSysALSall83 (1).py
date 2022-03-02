# Databricks notebook source
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pyarrow
import os

from tqdm import tqdm
import json
import sys

import glob

import time

import numba
from scipy import sparse as sp
from collections import defaultdict

import warnings
warnings.filterwarnings('ignore')

# from bayes_opt import BayesianOptimization

import random
from typing import Any, Dict

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV, cross_val_predict

import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
from scipy.optimize import minimize

pd.set_option("display.max_columns", 500)
pd.set_option("display.max_rows", 500)

print(pd.__version__)
print(pyarrow.__version__)

# COMMAND ----------

def average_precision(actual, recommended, k=30):
    ap_sum = 0
    hits = 0
    for i in range(k):
        product_id = recommended[i] if i < len(recommended) else None
        if product_id is not None and product_id in actual:
            hits += 1
            ap_sum += hits / (i + 1)
    return ap_sum / k

def normalized_average_precision(actual, recommended, k=30):
    actual = set(actual)
    if len(actual) == 0:
        return 0.0

    ap = average_precision(actual, recommended, k=k)
    ap_ideal = average_precision(actual, list(actual)[:k], k=k)
    return ap / ap_ideal

# COMMAND ----------

!pip install implicit
import implicit

# COMMAND ----------

# MAGIC %sh
# MAGIC wget -O azcopy.tar.gz https://aka.ms/downloadazcopy-v10-linux
# MAGIC tar -xf azcopy.tar.gz
# MAGIC cp "$(dirname "$(find . -path ./azcopy_linux\* -type f| tail -1)")"/azcopy azcopy
# MAGIC 
# MAGIC mkdir data
# MAGIC mkdir tovar
# MAGIC mkdir out
# MAGIC 
# MAGIC ./azcopy copy "https://bricksresult.blob.core.windows.net/data/json2/*?st=2020-03-04T06%3A58%3A48Z&se=2020-12-05T06%3A58%3A00Z&sp=racwdl&sv=2018-03-28&sr=c&sig=LFmu8D%2FVsGZfEj%2F45W6nldF2lvq2mQr9kFl6AblqUN8%3D"  "data/" 
# MAGIC 
# MAGIC ./azcopy copy "https://bricksresult.blob.core.windows.net/data/INVENTTABLE.parquet?st=2020-03-04T06%3A58%3A48Z&se=2020-12-05T06%3A58%3A00Z&sp=racwdl&sv=2018-03-28&sr=c&sig=LFmu8D%2FVsGZfEj%2F45W6nldF2lvq2mQr9kFl6AblqUN8%3D" "tovar/INVENTTABLE.parquet"

# COMMAND ----------

JSONS_DIR = './data'
BASE_SPLIT_POINT = "2019-03-01 10:05:00"

# COMMAND ----------

# MAGIC %run /Users/sale@rogainer.ru/src/utils

# COMMAND ----------

class ProductEncoder:
    def __init__(self, df):
        self.product_idx = {}
        self.product_pid = {}
        for idx, pid in enumerate(df.ITEMID.values):
            self.product_idx[pid] = idx
            self.product_pid[idx] = pid

    def toIdx(self, x):
        if type(x) == str:
            pid = x
            return self.product_idx[pid]
        return [self.product_idx[pid] for pid in x]

    def toPid(self, x):
        if type(x) == int:
            idx = x
            return self.product_pid[idx]
        return [self.product_pid[idx] for idx in x]

    @property
    def num_products(self):
        return len(self.product_idx)

# COMMAND ----------

df_tovar = pd.read_parquet('tovar/INVENTTABLE.parquet')
df_tovar.head()

# COMMAND ----------

product_encoder = ProductEncoder(pd.read_parquet('tovar/INVENTTABLE.parquet'))
product_encoder

# COMMAND ----------

def transaction_to_target(transaction: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "tid": transaction["tid"],
        "datetime": transaction["datetime"],
        "product_ids": [e["product_id"] for e in transaction["products"]],
        "store_id": transaction["store_id"],
    }


def get_client_info(client) -> Dict[str, Dict]:
    client_info = {}
    for row in client.itertuples():
        client_info[row.client_id] = {
            "age": row.age,
            "gender": row.gender,
            "client_id": row.client_id,
        }
    return client_info

# COMMAND ----------

# MAGIC %sh
# MAGIC ls data

# COMMAND ----------

def make_coo_row(transaction_history, product_encoder: ProductEncoder):
    idx = []
    values = []

    items = []
    for trans in transaction_history:
        items.extend([i["product_id"] for i in trans["products"]])
    n_items = len(items)

    for pid in items:
        if pid == '': 
            continue
        try: 
            idx.append(product_encoder.toIdx(pid))
        except KeyError:
            continue
        values.append(1.0)

    return sp.coo_matrix(
        (np.array(values).astype(np.float32), ([0] * len(idx), idx)), shape=(1, product_encoder.num_products),
    )

# COMMAND ----------



# COMMAND ----------

rows = []
for shard_id in range(50):
    for js in tqdm(json.loads(l) for l in open("{}/{:02d}.jsons".format(JSONS_DIR, shard_id))):
        rows.append(make_coo_row(js["transaction_history"], product_encoder))

# COMMAND ----------

X_sparse = sp.vstack(rows)
X_stored = X_sparse.tocsr()

# COMMAND ----------

os.environ["MKL_NUM_THREADS"] = "1"
model = implicit.als.AlternatingLeastSquares(factors=8, regularization=0.0, iterations=7)
model.fit(X_sparse.T)


# COMMAND ----------

# чистим память
import gc
del rows, X_sparse, js, X_stored
gc.collect()

# COMMAND ----------

m_ap = []
recomendations = {}
user_ids = []
prod_ids = []
scores = []
for shard_id in range(50):
    valid_data = [json.loads(l) for l in open("{}/{:02d}.jsons".format(JSONS_DIR, shard_id))]
    for i,js in enumerate(valid_data):
        row_sparse = make_coo_row(js["transaction_history"], product_encoder).tocsr()
        raw_recs = model.recommend(0, row_sparse, N=10, filter_already_liked_items=True, recalculate_user=True)
        recommended_items = product_encoder.toPid([x[0] for x in raw_recs])
#         recomendations[js['client_id']] = recommended_items

        user_ids += [js['client_id']] * len(recommended_items)
        prod_ids += recommended_items
        scores += [x[1] for x in raw_recs]


# COMMAND ----------

res = pd.DataFrame({'user_id':user_ids, 'prod_id':prod_ids, 'score':scores})

res.head()

# COMMAND ----------

del user_ids, prod_ids, scores
gc.collect()

# COMMAND ----------

res = res.merge(df_tovar[['ITEMID', 'ITEMNAME', 'ALK_EXTRADESC17ID']], how='left', left_on='prod_id', right_on='ITEMID')

# COMMAND ----------

len(res)

# COMMAND ----------

res.sample(5)

# COMMAND ----------

res_rec = res[res.ALK_EXTRADESC17ID == 'Активен к закупке']
res_rec.head()

# COMMAND ----------

res_rec.sample(8)

# COMMAND ----------

len(res_rec)

# COMMAND ----------

rec_tab = res_rec.groupby('user_id').head(7).reset_index(drop=True)
rec_tab.head()

# COMMAND ----------

rec_tab.sample(8)

# COMMAND ----------

rec_tab1 = rec_tab[['user_id', 'ITEMID','ITEMNAME','score']]

# COMMAND ----------

rec_tab1.head()

# COMMAND ----------

rec_tab1.tail()

# COMMAND ----------

rec_tab1.sample(20)

# COMMAND ----------

rec_tab1.to_parquet('out/rec_tab3als.parquet', index=False)

# COMMAND ----------

# проверка скорости выполнения частей функции:
# !pip install line_profiler
# %load_ext line_profiler
# %lprun -f fy fyn(valid_data)

# COMMAND ----------

# MAGIC %sh
# MAGIC ./azcopy copy "out/rec_tab3als.parquet" "https://bricksresult.blob.core.windows.net/data/rec_tab3als.parquet?st=2020-03-04T06%3A58%3A48Z&se=2020-12-05T06%3A58%3A00Z&sp=racwdl&sv=2018-03-28&sr=c&sig=LFmu8D%2FVsGZfEj%2F45W6nldF2lvq2mQr9kFl6AblqUN8%3D"   