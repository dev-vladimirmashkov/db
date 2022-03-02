# Databricks notebook source
from pyspark.sql import functions as F
from pyspark import StorageLevel

import pandas as pd
import numpy as np

# COMMAND ----------

from tqdm import tqdm
from scipy import sparse as sp

# COMMAND ----------

if any(mount.mountPoint == '/mnt/data' for mount in dbutils.fs.mounts()):
  dbutils.fs.unmount("/mnt/data")
if any(mount.mountPoint == '/mnt/out' for mount in dbutils.fs.mounts()):
  dbutils.fs.unmount("/mnt/out")

# COMMAND ----------

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

ClientHist = spark.read.format('parquet').options(
    header='true', inferschema='true').load("/mnt/data/dbo.CLIENT_HISTORY_2016.parquet")
CurTable = spark.read.format('parquet').options(
    header='true', inferschema='true').load("/mnt/data/dbo.ALK_DISCOUNTCARDTABLE.parquet")
DF = spark.read.format('parquet').options(
    header='true', inferschema='true').load("/mnt/data/dbo.ALK_DISCOUNTCARDTRANS.parquet")
OWNER = spark.read.format('parquet').options(
    header='true', inferschema='true').load("/mnt/data/dbo.OWNER.parquet")
Shops = spark.read.format('parquet').options(
    header='true', inferschema='true').load("/mnt/data/dbo.INVENTLOCATION.parquet")
Prod = spark.read.format('parquet').options(
    header='true', inferschema='true').load("/mnt/data/dbo.INVENTTABLE.parquet")

DF = DF.withColumn("client_id", F.col("DISCOUNTCARDTYPE") * 100000000 + F.col("DISCOUNTCARDID"))
OWNER = OWNER.withColumn("client_id", F.col("CURDISCOUNTCARDTYPE") * 100000000 + F.col("CURDISCOUNTCARDID"))
ClientHist = ClientHist.withColumn("client_id", F.col("CURDISCOUNTCARDTYPE") * 100000000 + F.col("CURDISCOUNTCARDID"))
CurTable = CurTable.withColumn("client_id", F.col("DISCOUNTCARDTYPE") * 100000000 + F.col("DISCOUNTCARDID"))
CurTable = CurTable.withColumn("CurClient_id", F.col("CURDISCOUNTCARDTYPE") * 100000000 + F.col("CURDISCOUNTCARDID"))

# OWNER = OWNER.persist(StorageLevel.MEMORY_AND_DISK)
# DF = DF.persist(StorageLevel.MEMORY_AND_DISK)
# Shops = Shops.persist(StorageLevel.MEMORY_AND_DISK)
# ClientHist = ClientHist.persist(StorageLevel.MEMORY_AND_DISK)

OWNER.createOrReplaceTempView("DF00")
DF.createOrReplaceTempView("DF11")
Shops.createOrReplaceTempView("DF22")
ClientHist.createOrReplaceTempView("DF33")
CurTable.createOrReplaceTempView("DF66")
Prod.createOrReplaceTempView("DF77")


# COMMAND ----------

display(DF)

# COMMAND ----------

display(CurTable)

# COMMAND ----------

# %sql
# SELECT user_id,
#   amount_all_life,
#   checks_all_life,
#   date_giveaway,
#   first_check_date,
#   last_check_date,
#   (DateDiff('2019-12-08', last_check_date)) LODold,
#   First_check_alk_shop,
#   LOD,
#   FOD,
#   Rcount,
#   OBLAST,
#   ShopName.INVENTLOCATIONID,
#   FirstOrderDate,
#   LastOrderDate,
#   Smax,
#   Smin,
#   ShopCount,
#   Qsum,
#   Amax,
#   Amean,
#   Astd,
#   Asum,
  
  
#   KOL_LET
#   ,BIRTHDATE
#   ,DATEREGISTERED
#   ,CITY
#   ,SEX

# FROM (SELECT COALESCE(Hist.client_id, Tec.client_id) AS user_id,*
# FROM DF33 Hist FULL JOIN
# (SELECT client_id
#       ,DISCOUNTCARDID
#       ,DISCOUNTCARDTYPE
      
#       ,MIN(DateDiff('2019-12-08', TRANSDATE)) LOD 
#       ,MAX(DateDiff('2019-12-08', TRANSDATE)) FOD 
#       ,COUNT(DISTINCT RECEIPTID) Rcount
#       ,MIN(TRANSDATE) FirstOrderDate 
#       ,MAX(TRANSDATE) LastOrderDate 
#       ,MAX(PopSh) Smax
#       ,MIN(PopSh) Smin
#       ,count(DISTINCT Trans.INVENTLOCATIONID) ShopCount
#       ,SUM(QTY) Qsum
#       ,MAX(AMOUNT) Amax
#       ,AVG(AMOUNT) Amean
#       ,STD(AMOUNT) Astd
#       ,SUM(AMOUNT) Asum      
 
#   FROM DF11 Trans LEFT JOIN (SELECT INVENTLOCATIONID,
#                                     count(DISTINCT RECEIPTID) PopSh
#                       FROM DF11
#                       GROUP BY INVENTLOCATIONID
#                       ) as PopShop
#                   ON Trans.INVENTLOCATIONID = PopShop.INVENTLOCATIONID
#   GROUP BY client_id
#        ,DISCOUNTCARDID
#       ,DISCOUNTCARDTYPE) Tec
#   ON  Hist.client_id = Tec.client_id)  AllCl
#   LEFT JOIN (SELECT client_id
#       ,KOL_LET
#       ,BIRTHDATE
#       ,DATEREGISTERED
#       ,CITY
#       ,SEX
  
#       FROM DF00) Ca
#   ON  AllCl.user_id = Ca.client_id 
#   LEFT JOIN (SELECT client_id
#                     ,SH.INVENTLOCATIONID

#                 FROM (SELECT client_id
#                     ,DISCOUNTCARDID
#                     ,DISCOUNTCARDTYPE
#                     ,INVENTLOCATIONID 
#                     ,ROW_NUMBER() OVER (PARTITION BY client_id ORDER BY (TRANSDATE)) Poryadok

#                     FROM DF11) as t LEFT JOIN DF22 SH
#                 ON t.INVENTLOCATIONID = SH.ALK_SHOPID
#                  WHERE t.Poryadok=1) ShopName
#   ON  AllCl.user_id = ShopName.client_id 
  
#   LEFT JOIN (SELECT client_id
#                     ,OBLAST
#               FROM(SELECT client_id
#                     ,OBLAST
#                     ,COUNT(DISTINCT RECEIPTID) 
#                     ,ROW_NUMBER() OVER (PARTITION BY client_id ORDER BY (COUNT(DISTINCT RECEIPTID)) DESC) Por   

#                 FROM DF11 TR LEFT JOIN DF22 SH
#                 ON TR.INVENTLOCATIONID = SH.ALK_SHOPID
#                 GROUP BY client_id
#                     ,OBLAST) PS
#               WHERE PS.Por=1) OBL
#   ON  AllCl.user_id = OBL.client_id
#   limit 500

# COMMAND ----------

DFres = spark.sql("""
SELECT CurClient_id
      ,CURDISCOUNTCARDID
      ,CURDISCOUNTCARDTYPE
      ,Trans.INVENTLOCATIONID
      ,Trans.CASSID
      ,Trans.TRANSDATE
      ,Trans.QTY
      ,Trans.AMOUNT
      ,Trans.ITEMID
      ,Trans.TRANSTIME
      ,Trans.TRANSDATETIME
      ,Trans.RECEIPTID 
FROM DF11 Trans LEFT JOIN DF66 
                  ON Trans.client_id = DF66.client_id
WHERE CurClient_id is not null
ORDER BY CurClient_id
  """)

# COMMAND ----------

display(DFres)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(DISTINCT CurClient_id)
# MAGIC 
# MAGIC FROM DF11 Trans LEFT JOIN DF66 
# MAGIC                   ON Trans.client_id = DF66.client_id
# MAGIC WHERE CurClient_id is not null

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

if any(mount.mountPoint == '/mnt/data1' for mount in dbutils.fs.mounts()):
  dbutils.fs.unmount("/mnt/data1")
dbutils.fs.mount(
source = "wasbs://data@bricksresult.blob.core.windows.net",
mount_point = "/mnt/data1",
extra_configs = {"fs.azure.sas.data.bricksresult.blob.core.windows.net":"?st=2020-03-04T06%3A58%3A48Z&se=2020-12-05T06%3A58%3A00Z&sp=racwdl&sv=2018-03-28&sr=c&sig=LFmu8D%2FVsGZfEj%2F45W6nldF2lvq2mQr9kFl6AblqUN8%3D"})

# COMMAND ----------

DFres.write.parquet('/mnt/data1/TransactionsCurClient/')

# COMMAND ----------

# Стереть дальше

# COMMAND ----------



# COMMAND ----------

rows = []
for client in tqdm(DFres.CurClient_id.values):
    idx = []
    values = []
    items = []    
    person = DFres[DFres.CurClient_id == client]
    for prod in person.ITEMID:
        items.append(prod)
    
    for pid in items:
        idx.append(product_encoder.toIdx(pid))
        values.append(person.loc[person.ITEMID == pid, 'QTY'].iloc[0])
    rows.append(sp.coo_matrix(
        (np.array(values).astype(np.float32), ([0] * len(idx), idx)), shape=(1, product_encoder.num_products),
    ))

# COMMAND ----------

DFres = spark.sql("""
SELECT user_id,
  amount_all_life,
  checks_all_life,
  first_check_date,
  last_check_date,
  (DateDiff('2016-01-01', last_check_date)) FODold,
  (DateDiff('2016-01-01', last_check_date)) LODold,
  First_check_alk_shop_u,
  LOD,
  FOD,
  Rcount,
  OBLAST,
  ShopName.INVENTLOCATIONID,
  FirstOrderDate,
  LastOrderDate,
  Smax,
  Smin,
  ShopCount,
  Qsum,
  Amax,
  Amean,
  Astd,
  Asum,
  (Q19/Q18) TrendQ,
  (A19/A18) TrendA,
  LETSales,
  
  KOL_LET_u
  ,BIRTHDATE_u
  ,BirthMonth_u
  ,DATEREGISTERED_u
  ,CITY_u
  ,SEX_u
  ,Ynov2018
  ,Ycol2018

FROM (SELECT COALESCE(Hist.CurClient_id, Tec.CurClient_id) AS user_id,*
FROM (SELECT CurClient_id,
                 SUM(amount_all_life) amount_all_life,
                SUM(checks_all_life) checks_all_life,
                MIN(first_check_date) first_check_date,
                MAX(last_check_date) last_check_date, 
                FIRST(First_check_alk_shop) First_check_alk_shop_u
                

FROM DF33 LEFT JOIN DF66
ON DF33.client_id = DF66.client_id

          LEFT JOIN DF22
ON DF33.First_check_alk_shop = DF22.INVENTLOCATIONID   

GROUP BY CurClient_id
       ) Hist FULL JOIN
(SELECT CurClient_id
      
      ,MIN(DateDiff('2019-12-09', TRANSDATE)) LOD 
      ,MAX(DateDiff('2019-12-09', TRANSDATE)) FOD 
      ,COUNT(DISTINCT RECEIPTID) Rcount
      ,MIN(TRANSDATE) FirstOrderDate 
      ,MAX(TRANSDATE) LastOrderDate 
      ,MAX(PopSh) Smax
      ,MIN(PopSh) Smin
      ,count(DISTINCT Trans.INVENTLOCATIONID) ShopCount
      ,SUM(QTY) Qsum
      ,MAX(AMOUNT) Amax
      ,AVG(AMOUNT) Amean
      ,STD(AMOUNT) Astd
      ,SUM(AMOUNT) Asum 
      
       
  FROM DF11 Trans LEFT JOIN (SELECT INVENTLOCATIONID,
                                    count(DISTINCT RECEIPTID) PopSh
                      FROM DF11
                      GROUP BY INVENTLOCATIONID
                      ) as PopShop
                  ON Trans.INVENTLOCATIONID = PopShop.INVENTLOCATIONID
                  LEFT JOIN DF66 
                  ON Trans.client_id = DF66.client_id
  WHERE TRANSDATE < '2019-12-09' and TRANSDATE >'2016-12-31'
  GROUP BY CurClient_id
       ) Tec
  ON  Hist.CurClient_id = Tec.CurClient_id)  AllCl
  LEFT JOIN (SELECT client_id
      ,FIRST(KOL_LET) KOL_LET_u
      ,FIRST(BIRTHDATE) BIRTHDATE_u
      ,MONTH(FIRST(BIRTHDATE)) BirthMonth_u
      ,FIRST(DATEREGISTERED) DATEREGISTERED_u
      ,FIRST(CITY) CITY_u
      ,FIRST(SEX) SEX_u
  
      FROM DF00
      GROUP BY client_id) Ca
  ON  AllCl.user_id = Ca.client_id 
  LEFT JOIN (SELECT client_id
                    ,SH.INVENTLOCATIONID

                FROM (SELECT client_id

                    ,INVENTLOCATIONID 
                    ,ROW_NUMBER() OVER (PARTITION BY client_id ORDER BY (TRANSDATE)) Poryadok

                    FROM DF11) as t LEFT JOIN DF22 SH
                ON t.INVENTLOCATIONID = SH.ALK_SHOPID
                 WHERE t.Poryadok=1) ShopName
  ON  AllCl.user_id = ShopName.client_id 
  
  LEFT JOIN (SELECT client_id
                    ,OBLAST
              FROM(SELECT client_id
                    ,OBLAST
                    ,COUNT(DISTINCT RECEIPTID) 
                    ,ROW_NUMBER() OVER (PARTITION BY client_id ORDER BY (COUNT(DISTINCT RECEIPTID)) DESC) Por   

                FROM DF11 TR LEFT JOIN DF22 SH
                ON TR.INVENTLOCATIONID = SH.ALK_SHOPID
                GROUP BY client_id
                    ,OBLAST) PS
              WHERE PS.Por=1) OBL
  ON  AllCl.user_id = OBL.client_id 
LEFT JOIN (SELECT CurClient_id,
                   1 Ynov2018
            FROM DF11 LEFT JOIN DF66
                      ON DF11.client_id = DF66.client_id
            WHERE TRANSDATE >= '2018-12-09' and DateDiff(TRANSDATE, '2018-12-09') < 60
            GROUP BY CurClient_id
  ) Ypredposl
ON  AllCl.user_id = Ypredposl.CurClient_id
   
  LEFT JOIN (SELECT CurClient_id,
                   1 Ycol2018
            FROM DF11 LEFT JOIN DF66
                      ON DF11.client_id = DF66.client_id
                      LEFT JOIN DF77
                      ON DF77.ITEMID = DF11.ITEMID
            WHERE TRANSDATE >= '2018-12-09' and DateDiff(TRANSDATE, '2018-12-09') < 60 
            and COLLECT = 'L_OREAL (ЛОРЕАЛЬ)' and CATEGOR = 'MAKE'
            GROUP BY CurClient_id
  ) YcolLET
ON  AllCl.user_id = YcolLET.CurClient_id  


LEFT JOIN (SELECT CurClient_id,
                   COUNT(DISTINCT RECEIPTID) Q19
            FROM DF11 LEFT JOIN DF66
                      ON DF11.client_id = DF66.client_id
                      
            WHERE TRANSDATE >= '2019-01-01' and TRANSDATE < '2019-12-09'
            GROUP BY CurClient_id
  ) Q2019
ON  AllCl.user_id = Q2019.CurClient_id

LEFT JOIN (SELECT CurClient_id,
                   COUNT(DISTINCT RECEIPTID) Q18
            FROM DF11 LEFT JOIN DF66
                      ON DF11.client_id = DF66.client_id
            WHERE TRANSDATE >= '2018-01-01' and TRANSDATE < '2018-12-09'
            GROUP BY CurClient_id
  ) Q2018
ON  AllCl.user_id = Q2018.CurClient_id

LEFT JOIN (SELECT CurClient_id,
                   SUM(AMOUNT) A19
            FROM DF11 LEFT JOIN DF66
                      ON DF11.client_id = DF66.client_id
            WHERE TRANSDATE >= '2019-01-01' and TRANSDATE < '2019-12-09'
            GROUP BY CurClient_id
  ) A2019
ON  AllCl.user_id = A2019.CurClient_id

LEFT JOIN (SELECT CurClient_id,
                   SUM(AMOUNT) A18
            FROM DF11 LEFT JOIN DF66
                      ON DF11.client_id = DF66.client_id
            WHERE TRANSDATE >= '2018-01-01' and TRANSDATE < '2018-12-09'
            GROUP BY CurClient_id
  ) A2018
ON  AllCl.user_id = A2018.CurClient_id

LEFT JOIN (SELECT CurClient_id,
                   COUNT(DISTINCT RECEIPTID) LETSales
            FROM DF11 LEFT JOIN DF66
                      ON DF11.client_id = DF66.client_id
                      LEFT JOIN DF77
                      ON DF77.ITEMID = DF11.ITEMID
             WHERE TRANSDATE < '2019-12-09' and TRANSDATE >'2016-12-31' 
            and COLLECT = 'L_OREAL (ЛОРЕАЛЬ)' and CATEGOR = 'MAKE'
            GROUP BY CurClient_id
  ) LetCount
ON  AllCl.user_id = LetCount.CurClient_id
WHERE LETSales > 2
  """)

# COMMAND ----------

DFres = DFres.fillna( { 'checks_all_life':0, 'Rcount':0 } )
DFres = DFres.withColumn("FirstShop", F.when(F.col('First_check_alk_shop_u').isNull(), F.col('ShopName.INVENTLOCATIONID')).otherwise(F.col('First_check_alk_shop_u')))
DFres = DFres.withColumn("FirstCheckDate", F.when(F.col('first_check_date').isNull(), F.col('FirstOrderDate')).otherwise(F.col('first_check_date')))
DFres = DFres.withColumn("LastCheckDate",
                         F.when(F.col('LastOrderDate').isNull(), F.col('last_check_date')).otherwise(F.col('LastOrderDate')))
DFres = DFres.withColumn("QOrders", F.col("checks_all_life")  + F.col("Rcount"))
DFres = DFres.withColumn("DavnostMin", F.least(F.col('LODold'), F.col('LOD')))
DFres = DFres.withColumn("DavnostMax", F.greatest(F.col('FOD'), F.col('FODold')))
DFres = DFres.withColumn("PNextOrder", F.round(F.col("QOrders") * 62 / (F.col("DavnostMax") + 62 - F.col("DavnostMin")), 2))
DFres = DFres.withColumn("SrChek", F.col("Asum")  / F.col("Rcount"))
DFres = DFres.withColumn("SrProd", F.col("Asum")  / F.col("Qsum"))
DFres = DFres.withColumn("SrDaysBetween", F.round((F.col("DavnostMax")  - F.col("DavnostMin")) / F.col("QOrders"), 2))

DFres = DFres.withColumn("Jdun", F.col("DavnostMin")  / F.col("SrDaysBetween"))

# COMMAND ----------

# %sql
# SELECT client_id,
#        MONTH(TRANSDATE) mesyac, 
#        COUNT(DISTINCT RECEIPTID) MSales
# FROM DF11
# WHERE TRANSDATE < '2018-11-01'
# GROUP BY client_id,
#           MONTH(TRANSDATE) 
# limit 500

# COMMAND ----------

SalesOborotnya = spark.sql("""
SELECT CurClient_id,
       MONTH(TRANSDATE) mesyac, 
       COUNT(DISTINCT RECEIPTID) MSales
FROM DF11  LEFT JOIN DF66 
                  ON DF11.client_id = DF66.client_id
WHERE TRANSDATE < '2019-12-09' and TRANSDATE >'2016-12-31'
  GROUP BY CurClient_id,
          MONTH(TRANSDATE) 
  """)

# COMMAND ----------

SalesOborotnya2 = spark.sql("""
SELECT CurClient_id as CurClient,
       (MONTH(TRANSDATE) + 100)  mesyacLET, 
       COUNT(DISTINCT RECEIPTID) MSales
FROM DF11  LEFT JOIN DF66 
                  ON DF11.client_id = DF66.client_id
            LEFT JOIN DF77
                      ON DF77.ITEMID = DF11.ITEMID
WHERE TRANSDATE < '2019-12-09' and TRANSDATE >'2016-12-31'
  and COLLECT = 'L_OREAL (ЛОРЕАЛЬ)' and CATEGOR = 'MAKE'
  GROUP BY CurClient_id,
          MONTH(TRANSDATE)
  """)

# COMMAND ----------

Matrica = SalesOborotnya.groupBy("CurClient_id").pivot("mesyac").sum("MSales").fillna(0)
DFres1 = DFres.join(Matrica, DFres.user_id == Matrica.CurClient_id, 'left')

# COMMAND ----------

Matrica2 = SalesOborotnya2.groupBy("CurClient").pivot("mesyacLET").sum("MSales").fillna(0)
DFres2 = DFres1.join(Matrica2, DFres1.user_id == Matrica2.CurClient, 'left')

# COMMAND ----------




# COMMAND ----------

DFres2 = DFres2.fillna( { 'KOL_LET_u':119 } )

# COMMAND ----------

# display(DFres3)

# COMMAND ----------

# DFres3 = DFres3.withColumn("TrendQ", F.col("2019")  / F.col("2018"))
# DFres3 = DFres3.withColumn("TrendA", F.col("sum2019")  / F.col("sum2018"))

# COMMAND ----------

XTable = DFres2.select('user_id', 'SrDaysBetween', 'KOL_LET_u', 'QOrders', 'SrProd', 
                               'SrChek', 'PNextOrder', 'DavnostMax', 'DavnostMin','BirthMonth_u', 'BIRTHDATE_u', "Jdun",
                             '1', '2', '3', '4', '5', '6',
                               '7', '8', '9', '10','11','12',
                              '101', '102', '103', '104', '105', '106',
                               '107', '108', '109', '110','111','112',
                                'OBLAST', 'SEX_u',
                              'Smax',  'Smin',  'ShopCount',  'Qsum',  'Amax',  'Amean',  'Astd',  'Asum',
                       'TrendQ', 'TrendA', 'LETSales',
                            'Ynov2018', 'Ycol2018'
                            )

# COMMAND ----------

# display(XTable)

# COMMAND ----------

# if any(mount.mountPoint == '/mnt/data1' for mount in dbutils.fs.mounts()):
#   dbutils.fs.unmount("/mnt/data1")
# dbutils.fs.mount(
# source = "wasbs://data@bricksresult.blob.core.windows.net",
# mount_point = "/mnt/data1",
# extra_configs = {"fs.azure.sas.data.bricksresult.blob.core.windows.net":"?st=2020-03-04T06%3A58%3A48Z&se=2020-12-05T06%3A58%3A00Z&sp=racwdl&sv=2018-03-28&sr=c&sig=LFmu8D%2FVsGZfEj%2F45W6nldF2lvq2mQr9kFl6AblqUN8%3D"})

# COMMAND ----------

XTable.write.parquet('/mnt/data1/df_test2019decColLOR_FE37/')

# COMMAND ----------

dbutils.fs.unmount("/mnt/data")
dbutils.fs.unmount("/mnt/data1")

# COMMAND ----------



# COMMAND ----------

