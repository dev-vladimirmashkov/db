# Databricks notebook source
from pyspark.sql import functions as F
from pyspark import StorageLevel

import pandas as pd
import numpy as np

# COMMAND ----------

from tqdm import tqdm
from scipy import sparse as sp
import os
import json
import sys

import glob

import random
from typing import Any, Dict
import hashlib

# COMMAND ----------

def md5_hash(x):
    return int(hashlib.md5(x.encode()).hexdigest(), 16)

# COMMAND ----------

if any(mount.mountPoint == '/mnt/data1' for mount in dbutils.fs.mounts()):
  dbutils.fs.unmount("/mnt/data1")
dbutils.fs.mount(
source = "wasbs://data@bricksresult.blob.core.windows.net",
mount_point = "/mnt/data1",
extra_configs = {"fs.azure.sas.data.bricksresult.blob.core.windows.net":"?st=2020-03-04T06%3A58%3A48Z&se=2020-12-05T06%3A58%3A00Z&sp=racwdl&sv=2018-03-28&sr=c&sig=LFmu8D%2FVsGZfEj%2F45W6nldF2lvq2mQr9kFl6AblqUN8%3D"})

# COMMAND ----------

DFres = spark.read.format('parquet').options(
    header='true', inferschema='true').load("/mnt/data1/TransactionsCurClient/")


# COMMAND ----------

display(DFres)

# COMMAND ----------

# MAGIC %sh
# MAGIC wget -O azcopy.tar.gz https://aka.ms/downloadazcopy-v10-linux
# MAGIC tar -xf azcopy.tar.gz
# MAGIC cp "$(dirname "$(find . -path ./azcopy_linux\* -type f| tail -1)")"/azcopy azcopy
# MAGIC 
# MAGIC mkdir out

# COMMAND ----------

spark_df = DFres.select(F.col("CurClient_id").alias("client_id"), F.col("RECEIPTID").alias("transaction_id"), F.col("ITEMID").alias("product_id"), F.col("INVENTLOCATIONID").alias("store_id")
                 , F.col("AMOUNT").alias("trn_sum_from_iss"), F.col("QTY").alias("product_quantity"), F.col("TRANSDATE").alias("transaction_datetime")).filter(F.col("trn_sum_from_iss") > 100)

# COMMAND ----------



# COMMAND ----------

output_jsons_dir = './out'
n_shards = 50 
output_path = output_jsons_dir
os.makedirs(
    output_path, exist_ok=True,
)
outs = []
for i in range(n_shards):
    outs.append(open(output_path + "/{:02d}.jsons".format(i), "w",))
_client = None
_transaction = None


columns = spark_df.schema.fieldNames()
chunks = spark_df.repartition(n_shards,"client_id").rdd.mapPartitions(lambda iterator: [pd.DataFrame(list(iterator), columns=columns)]).toLocalIterator()
for ii, df in enumerate(tqdm(chunks,total=n_shards)):
  df.transaction_datetime = df.transaction_datetime.astype(str)
  df.trn_sum_from_iss = df.trn_sum_from_iss.astype(float)
  df.product_quantity = df.product_quantity.astype(int)
  for row in df.itertuples():
    if _client is not None and _client["client_id"] != row.client_id:
      if _client is not None:
        _client["transaction_history"].append(_transaction)
        # rows are sharded by cliend_id
        shard_idx = md5_hash(str(_client["client_id"])) % n_shards
        data = _client
        outs[shard_idx].write(json.dumps(data) + "\n")

        _client = None
        _transaction = None

    if _client is None:
      _client = {
              "client_id": row.client_id,
              "transaction_history": [],
          }

    if _transaction is not None and _transaction["tid"] != row.transaction_id:
      _client["transaction_history"].append(_transaction)
      _transaction = None

    if _transaction is None:
      _transaction = {
                  "tid": row.transaction_id, "datetime": row.transaction_datetime, "products": [],
  #                 'rpr':row.regular_points_received,
  #                 'epr':row.express_points_received,
  #                 'rps':row.regular_points_spent,
  #                 'eps':row.express_points_spent,
  #                 'sum':row.purchase_sum,
                  'store_id':row.store_id,
          }

    p = {
              "product_id": row.product_id,
              "quantity": row.product_quantity,
              "s": row.trn_sum_from_iss,
  #             "r": "0" if row.trn_sum_from_red is None or pd.isna(row.trn_sum_from_red) else row.trn_sum_from_red,
          }
    _transaction["products"].append(p)
  if _client is not None:
    _client["transaction_history"].append(_transaction)
    # rows are sharded by cliend_id
    shard_idx = md5_hash(str(_client["client_id"])) % n_shards
    data = _client
    outs[shard_idx].write(json.dumps(data) + "\n")

    _client = None
    _transaction = None
for out in outs:
  out.close()  

# COMMAND ----------

# MAGIC %sh
# MAGIC ls out

# COMMAND ----------

# MAGIC %sh
# MAGIC ./azcopy copy "out/*" "https://bricksresult.blob.core.windows.net/data/json2/?st=2020-03-04T06%3A58%3A48Z&se=2020-12-05T06%3A58%3A00Z&sp=racwdl&sv=2018-03-28&sr=c&sig=LFmu8D%2FVsGZfEj%2F45W6nldF2lvq2mQr9kFl6AblqUN8%3D" 

# COMMAND ----------

