# Databricks notebook source
from pyspark.sql import functions as F
from pyspark import StorageLevel

import pandas as pd
import numpy as np

# COMMAND ----------

if any(mount.mountPoint == '/mnt/data1' for mount in dbutils.fs.mounts()):
  dbutils.fs.unmount("/mnt/data1")


# COMMAND ----------

configs = {"fs.azure.account.auth.type": "OAuth",
       "fs.azure.account.oauth.provider.type": "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider",
       "fs.azure.account.oauth2.client.id": "9bb009df-f8d0-472b-aadd-cda19252bd08",
       "fs.azure.account.oauth2.client.secret": "@g32sDu[b1:k=FgRAApHenG24DqlYNSw",
       "fs.azure.account.oauth2.client.endpoint": "https://login.microsoftonline.com/d986b6e6-6050-49f7-9e07-1cc9a3f74b2b/oauth2/token",
       "fs.azure.createRemoteFileSystemDuringInitialization": "true"}

dbutils.fs.mount(
source = "abfss://parquetfiles@alkordatalake.dfs.core.windows.net",
mount_point = "/mnt/data1",
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

# first(DF77.SALESCONTRIBUTIONRATIO), first(DF77.SALESMODEL),
# first(DF77.ALK_ITEMCOMMERSION),first(DF77.ALK_ALCOHOL), first(DF77.ALK_SEXID), first(DF77.ALK_ITEMCATEGOROLAPID), 

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

if any(mount.mountPoint == '/mnt/data1' for mount in dbutils.fs.mounts()):
  dbutils.fs.unmount("/mnt/data1")
dbutils.fs.mount(
source = "wasbs://data@bricksresult.blob.core.windows.net",
mount_point = "/mnt/data1",
extra_configs = {"fs.azure.sas.data.bricksresult.blob.core.windows.net":"?sv=2019-12-12&ss=bfqt&srt=sco&sp=rwdlacupx&se=2028-01-11T21:35:49Z&st=2021-01-12T13:35:49Z&spr=https&sig=%2F9iDTc%2FE8MOMfEAvfTYkTpil%2BR5fLzdLkBJyDbJsfWw%3D"})

# COMMAND ----------

# сохранить результат
DFres.write.parquet('/mnt/data/SalesLetoil/')

# COMMAND ----------

# MAGIC %sh
# MAGIC -fs ls /data

# COMMAND ----------

