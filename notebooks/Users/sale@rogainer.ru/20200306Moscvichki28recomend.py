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

import random
from typing import Any, Dict

# COMMAND ----------

JSONS_DIR = './out'
BASE_SPLIT_POINT = "2019-03-01 10:05:00"

# COMMAND ----------

# MAGIC %sh
# MAGIC wget -O azcopy.tar.gz https://aka.ms/downloadazcopy-v10-linux
# MAGIC tar -xf azcopy.tar.gz
# MAGIC cp "$(dirname "$(find . -path ./azcopy_linux\* -type f| tail -1)")"/azcopy azcopy
# MAGIC mkdir data
# MAGIC 
# MAGIC ./azcopy copy "https://bricksresult.blob.core.windows.net/data/moscviski28/*?st=2020-03-04T06%3A58%3A48Z&se=2020-12-05T06%3A58%3A00Z&sp=racwdl&sv=2018-03-28&sr=c&sig=LFmu8D%2FVsGZfEj%2F45W6nldF2lvq2mQr9kFl6AblqUN8%3D" data/  

# COMMAND ----------

# MAGIC %sh
# MAGIC ./azcopy copy "https://bricksresult.blob.core.windows.net/data/client/client08463274.parquet?st=2020-03-04T06%3A58%3A48Z&se=2020-12-05T06%3A58%3A00Z&sp=racwdl&sv=2018-03-28&sr=c&sig=LFmu8D%2FVsGZfEj%2F45W6nldF2lvq2mQr9kFl6AblqUN8%3D" data/client.parquet  

# COMMAND ----------

# df.to_parquet('./data/client.parquet', index=False)

# COMMAND ----------

# MAGIC %sh
# MAGIC ls -lah data

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

df = read_parquet_folder_as_pandas('data')
df.head()

# COMMAND ----------

df.head()

# COMMAND ----------

df.shape

# COMMAND ----------

df[df['DISCOUNTCARDID'] == '08463274']

# COMMAND ----------

df['client_id'] = 'D' + df.DISCOUNTCARDTYPE.astype(str) + df.DISCOUNTCARDID.astype(str)

# COMMAND ----------

clients = df.groupby(['client_id'])['TRANSDATE','RECEIPTID','KOL_LET','DISCOUNTCARDTYPE'].first().reset_index()
clients.columns = ['client_id','first_issue_date','first_redeem_date','age','gender']

clients.head()

# COMMAND ----------

clients.first_issue_date = clients.first_issue_date.astype(str)

# COMMAND ----------

purchases = df[['client_id', 'RECEIPTID', 'TRANSDATE','TRANSTIME','DISCOUNTCARDTYPE','DISCOUNTCARDID', 'DISCOUNTCARDTRANSTYPE'
               , 'AMOUNT', 'INVENTLOCATIONID', 'ITEMID', 'QTY', 'CASSID', 'TRANSDATETIME']]
purchases.columns = ['client_id','transaction_id','transaction_datetime','regular_points_received','express_points_received','regular_points_spent','express_points_spent'
                     ,'purchase_sum','store_id','product_id','product_quantity','trn_sum_from_iss','trn_sum_from_red']
purchases = purchases.sort_values(by=['client_id']).reset_index(drop=True)
purchases.head()

# COMMAND ----------

purchases.head()

# COMMAND ----------

baught = list(set(purchases[purchases.client_id == 'D0408463274'].product_id)) 

# COMMAND ----------

purchases.transaction_datetime = purchases.transaction_datetime.astype(str)
purchases.purchase_sum = purchases.purchase_sum.astype(str)
purchases.product_quantity = purchases.product_quantity.astype(int)

# COMMAND ----------

products = df.groupby(['ITEMID'])['client_id','RECEIPTID'].first().reset_index()
products.columns = ['product_id','level_1','level_2']
products.head()

# COMMAND ----------

# MAGIC %run /Users/sale@rogainer.ru/src/utils

# COMMAND ----------

class Transaction:
    def __init__(self, transaction_id, transaction_datetime, **kwargs):
        self.data = {
            **{"tid": transaction_id, "datetime": transaction_datetime, "products": [],},
            **kwargs,
        }

    def add_item(
        self, product_id: str, product_quantity: float, trn_sum_from_iss: float, trn_sum_from_red: float,
    ) -> None:
        p = {
            "product_id": product_id,
            "quantity": product_quantity,
            "s": trn_sum_from_iss,
            "r": "0" if trn_sum_from_red is None or pd.isna(trn_sum_from_red) else trn_sum_from_red,
        }
        self.data["products"].append(p)

    def as_dict(self,):
        return self.data

    def transaction_id(self,):
        return self.data["tid"]


class ClientHistory:
    def __init__(
        self, client_id,
    ):
        self.data = {
            "client_id": client_id,
            "transaction_history": [],
        }

    def add_transaction(
        self, transaction,
    ):
        self.data["transaction_history"].append(transaction)

    def as_dict(self,):
        return self.data

    def client_id(self,):
        return self.data["client_id"]


class RowSplitter:
    def __init__(
        self, output_path, n_shards=16,
    ):
        self.n_shards = n_shards
        os.makedirs(
            output_path, exist_ok=True,
        )
        self.outs = []
        for i in range(self.n_shards):
            self.outs.append(open(output_path + "/{:02d}.jsons".format(i), "w",))
        self._client = None
        self._transaction = None

    def finish(self,):
        self.flush()
        for outs in self.outs:
            outs.close()

    def flush(self,):
        if self._client is not None:
            self._client.add_transaction(self._transaction.as_dict())
            # rows are sharded by cliend_id
            shard_idx = md5_hash(str(self._client.client_id())) % self.n_shards
            data = self._client.as_dict()
            self.outs[shard_idx].write(json.dumps(data) + "\n")

            self._client = None
            self._transaction = None

    def consume_row(
        self, row,
    ):
        if self._client is not None and self._client.client_id() != row.client_id:
            self.flush()

        if self._client is None:
            self._client = ClientHistory(client_id=row.client_id)

        if self._transaction is not None and self._transaction.transaction_id() != row.transaction_id:
            self._client.add_transaction(self._transaction.as_dict())
            self._transaction = None

        if self._transaction is None:
            self._transaction = Transaction(
                transaction_id=row.transaction_id,
                transaction_datetime=row.transaction_datetime,
                rpr=row.regular_points_received,
                epr=row.express_points_received,
                rps=row.regular_points_spent,
                eps=row.express_points_spent,
                sum=row.purchase_sum,
                store_id=row.store_id,
            )

        self._transaction.add_item(
            product_id=row.product_id,
            product_quantity=row.product_quantity,
            trn_sum_from_iss=row.trn_sum_from_iss,
            trn_sum_from_red=row.trn_sum_from_red,
        )


def split_data_to_chunks(
    df, output_dir, n_shards
):
    splitter = RowSplitter(output_path=output_dir, n_shards=n_shards,)
    print("split_data_to_chunks: {} -> {}".format(products, output_dir,))
    for row in df.itertuples():
        splitter.consume_row(row)
    splitter.finish()


# COMMAND ----------

# MAGIC %sh
# MAGIC mkdir out

# COMMAND ----------

purchases.info()

# COMMAND ----------

output_jsons_dir = './out'

split_data_to_chunks(purchases, output_jsons_dir, n_shards=8)

# COMMAND ----------

# MAGIC %sh
# MAGIC ls -lah out

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

random.seed(43)  # lets be special

    
jsons_root = JSONS_DIR

client_info = get_client_info(clients)

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

        # copy info about client% client_id, age, gender
        sample = {**client_info[js["client_id"]]}
        sample["transaction_history"] = train_transactions
        sample["target"] = [transaction_to_target(x) for x in test_transactons]

        fout.write(json.dumps(sample) + "\n")
    fout.close()

# COMMAND ----------

import pandas as pd
import numpy as np
import json
from scipy import sparse as sp
from tqdm import tqdm
from collections import defaultdict

# COMMAND ----------

for i in range(8):
    for js in tqdm(json.loads(l) for l in open(get_shard_path(i))):
        if js['client_id'] == 'D0408463274':
            print(i)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



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

product_encoder = ProductEncoder(products)

# COMMAND ----------

rows = []
for shard_id in range(0,6):
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

# COMMAND ----------

svd = TruncatedSVD(n_components=128)
X_dense = svd.fit_transform(X_sparse)

# COMMAND ----------

from sklearn.neighbors import NearestNeighbors

# COMMAND ----------

num_neighbours = 512
knn = NearestNeighbors(n_neighbors=num_neighbours, metric="cosine")
knn.fit(X_dense)

# COMMAND ----------

m_ap = []
recomendations = {}
for js in tqdm(json.loads(l) for l in open(get_shard_path(6))):
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

rec = recomendations['D0408463274']

# COMMAND ----------

rec

# COMMAND ----------

baught = list(set(purchases[purchases.client_id == 'D0408463274'].product_id)) 

# COMMAND ----------

set(rec).intersection(set(baught))


# COMMAND ----------

len(set(rec) - set(rec).intersection(set(baught)))

# COMMAND ----------

pd.DataFrame(set(rec) - set(rec).intersection(set(baught)))

# COMMAND ----------

# MAGIC %sh
# MAGIC pip install implicit

# COMMAND ----------

import implicit

# COMMAND ----------

model = implicit.nearest_neighbours.CosineRecommender(K=10)
model.fit(X_sparse.T)

# COMMAND ----------

valid_data = [json.loads(l) for l in open(get_shard_path(6))]

# COMMAND ----------

m_ap = []
recomendations = {}
for js in tqdm(valid_data):
    row_sparse = make_coo_row(js["transaction_history"], product_encoder).tocsr()
    raw_recs = model.recommend(0, row_sparse, N=10, filter_already_liked_items=True, recalculate_user=True)
    recommended_items = product_encoder.toPid([x[0] for x in raw_recs])
    recomendations[js['client_id']] = recommended_items
    gt_items = js["target"][0]["product_ids"]
    m_ap.append(normalized_average_precision(gt_items, recommended_items, k=10))
print(np.mean(m_ap))

# COMMAND ----------

rec = recomendations['D0408463274']
len(set(rec) - set(rec).intersection(set(baught)))

# COMMAND ----------

pd.DataFrame(set(rec) - set(rec).intersection(set(baught)))

# COMMAND ----------

model = implicit.als.AlternatingLeastSquares(factors=32, regularization=0.0, iterations=8)
model.fit(X_sparse.T)

# COMMAND ----------

m_ap = []
recomendations = {}
for js in tqdm(valid_data):
    row_sparse = make_coo_row(js["transaction_history"], product_encoder).tocsr()
    raw_recs = model.recommend(0, row_sparse, N=20, filter_already_liked_items=True, recalculate_user=True)
    recommended_items = product_encoder.toPid([x[0] for x in raw_recs])
    recomendations[js['client_id']] = recommended_items
    gt_items = js["target"][0]["product_ids"]
    m_ap.append(normalized_average_precision(gt_items, recommended_items, k=20))
print(np.mean(m_ap))

# COMMAND ----------

rec = recomendations['D0408463274']
len(set(rec) - set(rec).intersection(set(baught)))

# COMMAND ----------

pd.DataFrame(set(rec) - set(rec).intersection(set(baught)))

# COMMAND ----------

