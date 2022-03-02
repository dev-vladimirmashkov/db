# Databricks notebook source
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

# COMMAND ----------

# Use the previously established DBFS mount point to read the data.
# create a data frame to read data.

DF = spark.read.format('parquet').options(
    header='true', inferschema='true').load("/mnt/data/dbo.OWNER.parquet")

# read the airline csv file and write the output to parquet format for easy query.

DF

# COMMAND ----------

DF.show()

# COMMAND ----------



# COMMAND ----------

DF1 = spark.read.format('parquet').options(
    header='true', inferschema='true').load("/mnt/data/dbo.ALK_DISCOUNTCARDTRANS.parquet")

# read the airline csv file and write the output to parquet format for easy query.

display(DF1)

# COMMAND ----------

DF.createOrReplaceTempView("DF00")

# COMMAND ----------

DF1.createOrReplaceTempView("DF11")

# COMMAND ----------

# MAGIC %sql SELECT * FROM DF00 limit 10

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT DISCOUNTCARDTRANSTYPE
# MAGIC       ,DISCOUNTCARDID
# MAGIC       ,DISCOUNTCARDTYPE
# MAGIC       ,INVENTLOCATIONID
# MAGIC       ,CASSID
# MAGIC       ,TRANSDATE
# MAGIC       ,QTY
# MAGIC       ,AMOUNT
# MAGIC       ,ITEMID
# MAGIC       ,TRANSTIME
# MAGIC       ,TRANSDATETIME
# MAGIC       ,RECEIPTID
# MAGIC 
# MAGIC   FROM DF11 Al INNER JOIN (SELECT CURDISCOUNTCARDID
# MAGIC       ,CURDISCOUNTCARDTYPE
# MAGIC       ,DATEREGISTERED
# MAGIC       ,FIRSTNAME
# MAGIC       ,LASTNAME
# MAGIC       ,PATRONYMICNAME
# MAGIC       ,BIRTHDATE
# MAGIC       ,ADDRESS
# MAGIC       ,PHONE
# MAGIC       ,EMAIL
# MAGIC       ,KOL_LET
# MAGIC   FROM DF00
# MAGIC   WHERE CITY = 'Г. МОСКВА' and KOL_LET > 37 and KOL_LET < 45 and SEX = 'Ж' ) Ca
# MAGIC   ON Al.DISCOUNTCARDID = Ca.CURDISCOUNTCARDID and Al.DISCOUNTCARDTYPE = Ca.CURDISCOUNTCARDTYPE
# MAGIC   WHERE MONTH(TRANSDATE)=3 and AMOUNT > 100 and DISCOUNTCARDTRANSTYPE = 1
# MAGIC   limit 10

# COMMAND ----------

dbutils.fs.mount(
source = "abfss://databricks@alkordatalake.dfs.core.windows.net",
mount_point = "/mnt/out",
extra_configs = configs)

# COMMAND ----------

DFres = spark.sql("""
SELECT DISCOUNTCARDTRANSTYPE
      ,DISCOUNTCARDID
      ,DISCOUNTCARDTYPE
      ,INVENTLOCATIONID
      ,CASSID
      ,TRANSDATE
      ,QTY
      ,AMOUNT
      ,ITEMID
      ,TRANSTIME
      ,TRANSDATETIME
      ,RECEIPTID

  FROM DF11 Al INNER JOIN (SELECT CURDISCOUNTCARDID
      ,CURDISCOUNTCARDTYPE
      ,DATEREGISTERED
      ,FIRSTNAME
      ,LASTNAME
      ,PATRONYMICNAME
      ,BIRTHDATE
      ,ADDRESS
      ,PHONE
      ,EMAIL
      ,KOL_LET
  FROM DF00
  WHERE CITY = 'Г. МОСКВА' and KOL_LET > 37 and KOL_LET < 45 and SEX = 'Ж' ) Ca
  ON Al.DISCOUNTCARDID = Ca.CURDISCOUNTCARDID and Al.DISCOUNTCARDTYPE = Ca.CURDISCOUNTCARDTYPE
  WHERE MONTH(TRANSDATE)=3 and AMOUNT > 100 and DISCOUNTCARDTRANSTYPE = 1
  limit 10""")


# COMMAND ----------

DFres.write.parquet('/mnt/out/res')

# COMMAND ----------

