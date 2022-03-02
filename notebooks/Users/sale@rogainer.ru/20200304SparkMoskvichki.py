# Databricks notebook source
if any(mount.mountPoint == '/mnt/data1' for mount in dbutils.fs.mounts()):
  dbutils.fs.unmount("/mnt/data1")
  

# COMMAND ----------

dbutils.fs.mount(
source = "wasbs://data@bricksresult.blob.core.windows.net",
mount_point = "/mnt/data1",
extra_configs = {"fs.azure.sas.data.bricksresult.blob.core.windows.net":"?st=2020-03-04T06%3A58%3A48Z&se=2020-12-05T06%3A58%3A00Z&sp=racwdl&sv=2018-03-28&sr=c&sig=LFmu8D%2FVsGZfEj%2F45W6nldF2lvq2mQr9kFl6AblqUN8%3D"})

# COMMAND ----------

# azcopy copy "https://alkordatalake.blob.core.windows.net/parquetfiles/dbo.ALK_DISCOUNTCARDTRANS.parquet" "https://bricksresult.blob.core.windows.net/data/ALK_DISCOUNTCARDTRANS.parquet?st=2020-03-04T06%3A58%3A48Z&se=2020-12-05T06%3A58%3A00Z&sp=racwdl&sv=2018-03-28&sr=c&sig=LFmu8D%2FVsGZfEj%2F45W6nldF2lvq2mQr9kFl6AblqUN8%3D"

# COMMAND ----------

# Use the previously established DBFS mount point to read the data.
# create a data frame to read data.

DF = spark.read.format('parquet').options(
    header='true', inferschema='true').load("/mnt/data1/OWNER.parquet")

# read the airline csv file and write the output to parquet format for easy query.

DF

# COMMAND ----------

display(DF)

# COMMAND ----------

DF1 = spark.read.format('parquet').options(
    header='true', inferschema='true').load("/mnt/data1/ALK_DISCOUNTCARDTRANS.parquet")

# read the airline csv file and write the output to parquet format for easy query.

display(DF1)

# COMMAND ----------

DF.createOrReplaceTempView("DF00")

# COMMAND ----------

DF1.createOrReplaceTempView("DF11")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT count(*)
# MAGIC   FROM DF11

# COMMAND ----------

# MAGIC %sql SELECT * FROM DF00 limit 10

# COMMAND ----------

# MAGIC %sql
# MAGIC -- SELECT DISCOUNTCARDTRANSTYPE
# MAGIC --       ,DISCOUNTCARDID
# MAGIC --       ,DISCOUNTCARDTYPE
# MAGIC --       ,INVENTLOCATIONID
# MAGIC --       ,CASSID
# MAGIC --       ,TRANSDATE
# MAGIC --       ,QTY
# MAGIC --       ,AMOUNT
# MAGIC --       ,ITEMID
# MAGIC --       ,TRANSTIME
# MAGIC --       ,TRANSDATETIME
# MAGIC --       ,RECEIPTID
# MAGIC 
# MAGIC --   FROM DF11 Al INNER JOIN (SELECT CURDISCOUNTCARDID
# MAGIC --       ,CURDISCOUNTCARDTYPE
# MAGIC --       ,DATEREGISTERED
# MAGIC --       ,FIRSTNAME
# MAGIC --       ,LASTNAME
# MAGIC --       ,PATRONYMICNAME
# MAGIC --       ,BIRTHDATE
# MAGIC --       ,ADDRESS
# MAGIC --       ,PHONE
# MAGIC --       ,EMAIL
# MAGIC --       ,KOL_LET
# MAGIC --   FROM DF00
# MAGIC --   WHERE CITY = 'Г. МОСКВА' and KOL_LET > 28 and KOL_LET < 37 and SEX = 'Ж' ) Ca
# MAGIC --   ON Al.DISCOUNTCARDID = Ca.CURDISCOUNTCARDID and Al.DISCOUNTCARDTYPE = Ca.CURDISCOUNTCARDTYPE
# MAGIC --   WHERE MONTH(TRANSDATE)=3 and AMOUNT > 100 and DISCOUNTCARDTRANSTYPE = 1

# COMMAND ----------

# dbutils.fs.mount(
# source = "abfss://databricks@alkordatalake.dfs.core.windows.net",
# mount_point = "/mnt/moscviski35",
# extra_configs = configs)

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
      ,KOL_LET
      
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
  WHERE CITY = 'Г. МОСКВА' and KOL_LET > 28 and KOL_LET < 38 and SEX = 'Ж' ) Ca
  ON Al.DISCOUNTCARDID = Ca.CURDISCOUNTCARDID and Al.DISCOUNTCARDTYPE = Ca.CURDISCOUNTCARDTYPE
  WHERE AMOUNT > 100 and DISCOUNTCARDTRANSTYPE = 1 and ITEMID is not null
  """)


# COMMAND ----------

# display(DFres)

# COMMAND ----------

DFres.write.parquet('/mnt/data1/moscviski35/full/')

# COMMAND ----------

