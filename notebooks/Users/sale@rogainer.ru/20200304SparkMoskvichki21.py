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

# MAGIC %sql SELECT * FROM DF00 limit 10

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
  WHERE CITY = 'Г. МОСКВА' and KOL_LET > 21 and KOL_LET < 29 and SEX = 'Ж' ) Ca
  ON Al.DISCOUNTCARDID = Ca.CURDISCOUNTCARDID and Al.DISCOUNTCARDTYPE = Ca.CURDISCOUNTCARDTYPE
  WHERE AMOUNT > 100 and DISCOUNTCARDTRANSTYPE = 1 and ITEMID is not null
  """)


# COMMAND ----------

# display(DFres)

# COMMAND ----------

DFres.write.parquet('/mnt/data1/moscviski28/')

# COMMAND ----------

