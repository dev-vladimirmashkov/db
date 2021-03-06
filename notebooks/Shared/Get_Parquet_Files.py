# Databricks notebook source
from pyspark.sql import functions as F
from pyspark import StorageLevel

import pandas as pd
import numpy as np

# COMMAND ----------

if any(mount.mountPoint == '/mnt/data' for mount in dbutils.fs.mounts()):
  dbutils.fs.unmount("/mnt/data")


# COMMAND ----------

configs = {"fs.azure.account.auth.type": "OAuth",
       "fs.azure.account.oauth.provider.type": "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider",
       "fs.azure.account.oauth2.client.id": "9bb009df-f8d0-472b-aadd-cda19252bd08",
       "fs.azure.account.oauth2.client.secret": "@g32sDu[b1:k=FgRAApHenG24DqlYNSw",
       "fs.azure.account.oauth2.client.endpoint": "https://login.microsoftonline.com/d986b6e6-6050-49f7-9e07-1cc9a3f74b2b/oauth2/token",
       "fs.azure.createRemoteFileSystemDuringInitialization": "true"}

dbutils.fs.mount(
source = "abfss://letoile@alkordatalake.dfs.core.windows.net/raw/ml",
mount_point = "/mnt/data",
extra_configs = configs)

# COMMAND ----------

DF = spark.read.format('parquet').options(
    header='true', inferschema='true').load("/mnt/data/dbo.RetailTransactionSalesTrans.parquet")
Prod = spark.read.format('parquet').options(
    header='true', inferschema='true').load("/mnt/data/dbo.INVENTTABLE.parquet")


DF.createOrReplaceTempView("DF11")
Prod.createOrReplaceTempView("DF77")

# COMMAND ----------

display(DF)

# COMMAND ----------

tmp = spark.sql("select ALK_SEXID from DF77")

# COMMAND ----------

# todo написать селект с временными рядами продаж товаров ITEMID (где COLLECT = 'L`ETOILE (Л`ЭТУАЛЬ)') по месяцам 
DFres = spark.sql("""
SELECT Trans.ITEMID, YEAR(TRANSDATE) year, MONTH(TRANSDATE) month, -sum(QTY) salesQ, -sum(NETAMOUNT) salesM, DF77.ALK_SEXID id_sex
FROM DF11 Trans LEFT JOIN DF77 
ON Trans.ITEMID = DF77.ITEMID
WHERE DF77.COLLECT = 'L`ETOILE (Л`ЭТУАЛЬ)' 
GROUP BY  Trans.ITEMID, YEAR(TRANSDATE), MONTH(TRANSDATE), id_sex
ORDER BY Trans.ITEMID, year, month """ )

# COMMAND ----------

# MAGIC %sh
# MAGIC wget -O azcopy.tar.gz https://aka.ms/downloadazcopy-v10-linux
# MAGIC tar -xf azcopy.tar.gz
# MAGIC cp "$(dirname "$(find . -path ./azcopy_linux\* -type f| tail -1)")"/azcopy azcopy
# MAGIC mkdir data
# MAGIC mkdir model
# MAGIC ./azcopy remove "https://alkordatalake.dfs.core.windows.net/letoile/raw/ml/SalesLetoil?sv=2020-02-10&st=2020-12-31T21%3A00%3A00Z&se=2050-03-31T20%3A59%3A00Z&sr=c&sp=racwdlme&sig=69jYjI73deMsJohT8L9aDGjPE3k9ohHjdzmPPPkTD1s%3D" --recursive --trusted-microsoft-suffixes= --log-level=INFO;

# COMMAND ----------

DFres.write.parquet('/mnt/data/SalesLetoil/')