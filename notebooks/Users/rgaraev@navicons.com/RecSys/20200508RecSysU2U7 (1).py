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

# MAGIC %sh
# MAGIC wget -O azcopy.tar.gz https://aka.ms/downloadazcopy-v10-linux
# MAGIC tar -xf azcopy.tar.gz
# MAGIC cp "$(dirname "$(find . -path ./azcopy_linux\* -type f| tail -1)")"/azcopy azcopy
# MAGIC 
# MAGIC mkdir data
# MAGIC mkdir tovar
# MAGIC mkdir out
# MAGIC 
# MAGIC ./azcopy copy "https://bricksresult.blob.core.windows.net/data/json2/03.jsons?st=2020-03-04T06%3A58%3A48Z&se=2020-12-05T06%3A58%3A00Z&sp=racwdl&sv=2018-03-28&sr=c&sig=LFmu8D%2FVsGZfEj%2F45W6nldF2lvq2mQr9kFl6AblqUN8%3D"  "data/03.jsons" 
# MAGIC ./azcopy copy "https://bricksresult.blob.core.windows.net/data/json2/02.jsons?st=2020-03-04T06%3A58%3A48Z&se=2020-12-05T06%3A58%3A00Z&sp=racwdl&sv=2018-03-28&sr=c&sig=LFmu8D%2FVsGZfEj%2F45W6nldF2lvq2mQr9kFl6AblqUN8%3D"  "data/02.jsons" 
# MAGIC ./azcopy copy "https://bricksresult.blob.core.windows.net/data/json2/01.jsons?st=2020-03-04T06%3A58%3A48Z&se=2020-12-05T06%3A58%3A00Z&sp=racwdl&sv=2018-03-28&sr=c&sig=LFmu8D%2FVsGZfEj%2F45W6nldF2lvq2mQr9kFl6AblqUN8%3D"  "data/01.jsons"  
# MAGIC ./azcopy copy "https://bricksresult.blob.core.windows.net/data/json2/00.jsons?st=2020-03-04T06%3A58%3A48Z&se=2020-12-05T06%3A58%3A00Z&sp=racwdl&sv=2018-03-28&sr=c&sig=LFmu8D%2FVsGZfEj%2F45W6nldF2lvq2mQr9kFl6AblqUN8%3D"  "data/00.jsons" 
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
# MAGIC ./azcopy copy "https://bricksresult.blob.core.windows.net/data/OWNER.parquet?st=2020-03-04T06%3A58%3A48Z&se=2020-12-05T06%3A58%3A00Z&sp=racwdl&sv=2018-03-28&sr=c&sig=LFmu8D%2FVsGZfEj%2F45W6nldF2lvq2mQr9kFl6AblqUN8%3D" "tovar/OWNER.parquet"

# COMMAND ----------

# df_client = pd.read_parquet('tovar/OWNER.parquet')

# COMMAND ----------

# df_client.sample(5)

# COMMAND ----------

# df_client['client_id'] = df_client.CURDISCOUNTCARDTYPE.astype(str) + df_client.CURDISCOUNTCARDID.astype(str)

# COMMAND ----------



# COMMAND ----------

# clients = df_client.groupby(['client_id'])['DATEREGISTERED','BIRTHDATE','KOL_LET','SEX'].first().reset_index()

# COMMAND ----------

# clients.columns = ['client_id','first_issue_date','first_redeem_date','age','gender']

# COMMAND ----------

# clients.head()

# COMMAND ----------

# clients.sample(5)

# COMMAND ----------

# clients.first_issue_date = clients.first_issue_date.astype(str)

# COMMAND ----------

random.seed(42)  # lets be special

    
jsons_root = JSONS_DIR

# client_info = get_client_info(clients)

print("process shards")
for js_path in tqdm(sorted(glob.glob(jsons_root + "/*.jsons"))):
    fout = open(js_path + ".splitted", "w")
    for js in (json.loads(s) for s in open(js_path)):
        sorted_transactions = sorted(js["transaction_history"], key=lambda x: x["datetime"])
        split_candidates = [
            t["datetime"] for t in sorted_transactions if t["datetime"] > BASE_SPLIT_POINT
        ]
        if len(split_candidates) == 0:
            # no transactions after split points - so we cannot validates on this sample, skip it.
            continue
        split_point = random.choice(split_candidates)
        train_transactions = [t for t in sorted_transactions if t["datetime"] < split_point]
        test_transactons = [t for t in sorted_transactions if t["datetime"] >= split_point]

        # copy info about client% client_id, age, gender (пока не надо)
        sample = {
                        "client_id": js["client_id"],
        }
        sample["transaction_history"] = train_transactions
        sample["target"] = [transaction_to_target(x) for x in test_transactons]

        fout.write(json.dumps(sample) + "\n")
    fout.close()

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
        idx.append(product_encoder.toIdx(pid))
        values.append(1.0 / n_items)

    return sp.coo_matrix(
        (np.array(values).astype(np.float32), ([0] * len(idx), idx)), shape=(1, product_encoder.num_products),
    )

# COMMAND ----------

rows = []
for shard_id in range(2):
    for js in tqdm(json.loads(l) for l in open(get_shard_path(shard_id))):
        rows.append(make_coo_row(js["transaction_history"], product_encoder))

# COMMAND ----------

X_sparse = sp.vstack(rows)

# COMMAND ----------

X_sparse.shape

# COMMAND ----------

X_stored = X_sparse.tocsr()

# COMMAND ----------

from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=128)
X_dense = svd.fit_transform(X_sparse)

# COMMAND ----------

from sklearn.neighbors import NearestNeighbors
num_neighbours = 16
knn = NearestNeighbors(n_neighbors=num_neighbours, 
#                        n_jobs=-1, 
                       metric="cosine")
knn.fit(X_dense)

# COMMAND ----------

# 5 минут на 1000 итераций - num_neighbours = 32 - слишком медленно
# 5 минут на 1000 итераций - num_neighbours = 512 - слишком медленно
m_ap = []
recomendations = {}
for js in tqdm(json.loads(l) for l in open(get_shard_path(3))):
    row_sparse = make_coo_row(js["transaction_history"], product_encoder)
    row_dense = svd.transform(row_sparse)
    knn_result = knn.kneighbors(row_dense, n_neighbors=num_neighbours)
    neighbors = knn_result[1]
    scores = np.asarray(X_stored[neighbors[0]].sum(axis=0)[0]).flatten()
    top_indices = np.argsort(-scores)
    recommended_items = product_encoder.toPid(top_indices[:10])
    recomendations[js['client_id']] = recommended_items
    gt_items = js["target"][0]["product_ids"]
    m_ap.append(normalized_average_precision(gt_items, recommended_items, k=10))
print(np.mean(m_ap))

# COMMAND ----------



# COMMAND ----------

