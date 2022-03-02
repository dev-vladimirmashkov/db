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

from pyspark.sql import functions as F

# COMMAND ----------

if any(mount.mountPoint == '/mnt/data1' for mount in dbutils.fs.mounts()):
  dbutils.fs.unmount("/mnt/data1")
dbutils.fs.mount(
source = "wasbs://data@bricksresult.blob.core.windows.net",
mount_point = "/mnt/data1",
extra_configs = {"fs.azure.sas.data.bricksresult.blob.core.windows.net":"?st=2020-03-04T06%3A58%3A48Z&se=2020-12-05T06%3A58%3A00Z&sp=racwdl&sv=2018-03-28&sr=c&sig=LFmu8D%2FVsGZfEj%2F45W6nldF2lvq2mQr9kFl6AblqUN8%3D"})

# COMMAND ----------

res_rec = spark.read.format('parquet').options(
    header='true', inferschema='true').load("/mnt/data1/rec_tab2all.parquet")
res_rec1 = spark.read.format('parquet').options(
    header='true', inferschema='true').load("/mnt/data1/rec_tab3als.parquet")

# COMMAND ----------

display(res_rec)

# COMMAND ----------

if any(mount.mountPoint == '/mnt/data' for mount in dbutils.fs.mounts()):
  dbutils.fs.unmount("/mnt/data")
configs = {"fs.azure.account.auth.type": "OAuth",
       "fs.azure.account.oauth.provider.type": "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider",
       "fs.azure.account.oauth2.client.id": "9bb009df-f8d0-472b-aadd-cda19252bd08",
       "fs.azure.account.oauth2.client.secret": "@g32sDu[b1:k=FgRAApHenG24DqlYNSw",
       "fs.azure.account.oauth2.client.endpoint": "https://login.microsoftonline.com/d986b6e6-6050-49f7-9e07-1cc9a3f74b2b/oauth2/token",
       "fs.azure.createRemoteFileSystemDuringInitialization": "true"}

dbutils.fs.mount(
source = "abfss://parquetfiles@alkordatalake.dfs.core.windows.net",
mount_point = "/mnt/data",
extra_configs = configs)
CurTable = spark.read.format('parquet').options(
    header='true', inferschema='true').load("/mnt/data/dbo.ALK_DISCOUNTCARDTABLE.parquet")
CurTable = CurTable.withColumn("CurClient_id", F.col("CURDISCOUNTCARDTYPE") * 100000000 + F.col("CURDISCOUNTCARDID"))
CurTable.createOrReplaceTempView("DF66")

# COMMAND ----------

res_rec.createOrReplaceTempView("DF00")
res_rec1.createOrReplaceTempView("DF11")

# COMMAND ----------

DFres = spark.sql("""
SELECT CurClient_id,
        CURDISCOUNTCARDID,
        CURDISCOUNTCARDTYPE,
        ITEMID,
        ITEMNAME,
        score,
        'ALS' MODEL
FROM DF11 LEFT JOIN DF66
ON DF11.user_id = DF66.CurClient_id
GROUP BY CurClient_id,
        CURDISCOUNTCARDID,
        CURDISCOUNTCARDTYPE,
        ITEMID,
        ITEMNAME,
        score
UNION ALL
SELECT CurClient_id,
        CURDISCOUNTCARDID,
        CURDISCOUNTCARDTYPE,
        ITEMID,
        ITEMNAME,
        score,
        'I2I' MODEL
FROM DF00 LEFT JOIN DF66
ON DF00.user_id = DF66.CurClient_id
GROUP BY CurClient_id,
        CURDISCOUNTCARDID,
        CURDISCOUNTCARDTYPE,
        ITEMID,
        ITEMNAME,
        score
""")

# COMMAND ----------

display(DFres)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT CurClient_id,
# MAGIC         CURDISCOUNTCARDID,
# MAGIC         CURDISCOUNTCARDTYPE,
# MAGIC         ITEMID,
# MAGIC         ITEMNAME,
# MAGIC         score
# MAGIC FROM DF11 LEFT JOIN DF66
# MAGIC ON DF11.user_id = DF66.CurClient_id
# MAGIC WHERE user_id = 402853296
# MAGIC GROUP BY CurClient_id,
# MAGIC         CURDISCOUNTCARDID,
# MAGIC         CURDISCOUNTCARDTYPE,
# MAGIC         ITEMID,
# MAGIC         ITEMNAME,
# MAGIC         score

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT CurClient_id,
# MAGIC         CURDISCOUNTCARDID,
# MAGIC         CURDISCOUNTCARDTYPE,
# MAGIC         ITEMID,
# MAGIC         ITEMNAME,
# MAGIC         score
# MAGIC FROM DF11 LEFT JOIN DF66
# MAGIC ON DF11.user_id = DF66.CurClient_id
# MAGIC WHERE user_id = 505259042
# MAGIC GROUP BY CurClient_id,
# MAGIC         CURDISCOUNTCARDID,
# MAGIC         CURDISCOUNTCARDTYPE,
# MAGIC         ITEMID,
# MAGIC         ITEMNAME,
# MAGIC         score

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT CurClient_id,
# MAGIC         CURDISCOUNTCARDID,
# MAGIC         CURDISCOUNTCARDTYPE,
# MAGIC         ITEMID,
# MAGIC         ITEMNAME,
# MAGIC         score
# MAGIC FROM DF11 LEFT JOIN DF66
# MAGIC ON DF11.user_id = DF66.CurClient_id
# MAGIC WHERE user_id = 407257304
# MAGIC GROUP BY CurClient_id,
# MAGIC         CURDISCOUNTCARDID,
# MAGIC         CURDISCOUNTCARDTYPE,
# MAGIC         ITEMID,
# MAGIC         ITEMNAME,
# MAGIC         score

# COMMAND ----------



# COMMAND ----------

dwDatabase = "AlkorSynapsePool"
dwServer = "alkor-dwh-server.database.windows.net"
dwUser = "AlkorLoader"
dwPass = "Navicon999"
dwJdbcPort =  "1433"
dwJdbcExtraOptions = "encrypt=true;trustServerCertificate=true;hostNameInCertificate=*.database.windows.net;loginTimeout=30;"
sqlDwUrl = "jdbc:sqlserver://" + dwServer + ":" + dwJdbcPort + ";database=" + dwDatabase + ";user=" + dwUser+";password=" + dwPass + ";$dwJdbcExtraOptions"
sqlDwUrlSmall = "jdbc:sqlserver://" + dwServer + ":" + dwJdbcPort + ";database=" + dwDatabase + ";user=" + dwUser+";password=" + dwPass

# COMMAND ----------

spark.conf.set(
  "fs.azure.account.key.alkordatalake.dfs.core.windows.net",
  "F+S/Qo4A/ExOAmhwhmtll28Jj5e24l60/0nCxd9jHjTKax1cV0EiNHBJ52zE6UYpSvWgHu3kHPQjV4gW4qjtcw==")

# COMMAND ----------

spark.conf.set("fs.azure.createRemoteFileSystemDuringInitialization", "true")
dbutils.fs.ls("abfss://bricksresult@alkordatalake.dfs.core.windows.net")
spark.conf.set("fs.azure.createRemoteFileSystemDuringInitialization", "false")

# COMMAND ----------

tempDir = "abfss://bricksresult@alkordatalake.dfs.core.windows.net/tempDirs"

# COMMAND ----------

spark.conf.set(
    "spark.sql.parquet.writeLegacyFormat",
    "true")

DFres.write.format("com.databricks.spark.sqldw").option("url", sqlDwUrlSmall).option("dbtable", "RecomendTable").option( "forward_spark_azure_storage_credentials",
                                                                                                                    "True").option("tempdir", tempDir).mode("overwrite").save()

# COMMAND ----------

