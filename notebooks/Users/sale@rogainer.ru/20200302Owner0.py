# Databricks notebook source
import pandas as pd
import numpy as np

# COMMAND ----------


# wget -O azcopy.tar.gz https://aka.ms/downloadazcopy-v10-linux
# tar -xf azcopy.tar.gz
# cp "$(dirname "$(find . -path ./azcopy_linux\* -type f| tail -1)")"/azcopy azcopy
# mkdir data
# ./azcopy login --tenant-id "d986b6e6-6050-49f7-9e07-1cc9a3f74b2b"
# ./azcopy copy "https://alkordatalake.blob.core.windows.net/parquetfiles/dbo.INVENTDIM.parquet" data/

# COMMAND ----------

# MAGIC %sh
# MAGIC wget -O azcopy.tar.gz https://aka.ms/downloadazcopy-v10-linux
# MAGIC tar -xf azcopy.tar.gz
# MAGIC cp "$(dirname "$(find . -path ./azcopy_linux\* -type f| tail -1)")"/azcopy azcopy
# MAGIC mkdir data
# MAGIC 
# MAGIC 
# MAGIC export AZCOPY_SPA_CLIENT_SECRET="@g32sDu[b1:k=FgRAApHenG24DqlYNSw"
# MAGIC 
# MAGIC ./azcopy login --service-principal --application-id 9bb009df-f8d0-472b-aadd-cda19252bd08 --tenant-id "d986b6e6-6050-49f7-9e07-1cc9a3f74b2b"
# MAGIC ./azcopy copy "https://alkordatalake.blob.core.windows.net/parquetfiles/dbo.INVENTDIM.parquet" data/

# COMMAND ----------

dbutils.secrets.get(scope = "Mike", key = "drug")

# COMMAND ----------

df = pd.read_parquet('data/dbo.INVENTDIM.parquet')
df.head()

# COMMAND ----------

df1 = df.head()

# COMMAND ----------

df1

# COMMAND ----------

df1.to_parquet('data/test1.parquet')

# COMMAND ----------

# MAGIC %sh
# MAGIC ls data

# COMMAND ----------

# MAGIC %sh
# MAGIC 
# MAGIC ./azcopy copy data/test1.parquet "https://bricksresult.blob.core.windows.net/result/test1.parquet?st=2020-03-03T14%3A46%3A05Z&se=2222-12-21T21%3A00%3A00Z&sp=racwdl&sv=2018-03-28&sr=c&sig=O62q2uFN52ysf%2FeJmlzG1PzOpR1ZxKkIo3e0iNMs%2FM8%3D" 

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------


# apt-get install keyutils -y

# COMMAND ----------

# MAGIC %sh
# MAGIC sudo apt install keyutils

# COMMAND ----------

# MAGIC %sh
# MAGIC sudo keyctl new_session

# COMMAND ----------

# MAGIC %sh
# MAGIC keyctl show

# COMMAND ----------

# MAGIC %sh
# MAGIC lsb_release -a

# COMMAND ----------

# MAGIC %sh
# MAGIC 
# MAGIC sudo keyctl new_session
# MAGIC 
# MAGIC export AZCOPY_SPA_CLIENT_SECRET=@g32sDu[b1:k=FgRAApHenG24DqlYNSw
# MAGIC sudo -E bash -c "echo $AZCOPY_SPA_CLIENT_SECRET"
# MAGIC ./azcopy login --service-principal --application-id 9bb009df-f8d0-472b-aadd-cda19252bd08 --tenant-id d986b6e6-6050-49f7-9e07-1cc9a3f74b2b
# MAGIC keyctl show
# MAGIC ./azcopy copy data/test1.parquet "https://alkordatalake.blob.core.windows.net/databricks/test1.parquet" 

# COMMAND ----------

# MAGIC %sh
# MAGIC sudo -E bash -c "echo $AZCOPY_SPA_CLIENT_SECRET"

# COMMAND ----------

# MAGIC %sh
# MAGIC keyctl session workaroundSession
# MAGIC export AZCOPY_SPA_CLIENT_SECRET="@g32sDu[b1:k=FgRAApHenG24DqlYNSw"
# MAGIC ./azcopy login --service-principal --application-id 9bb009df-f8d0-472b-aadd-cda19252bd08 --tenant-id d986b6e6-6050-49f7-9e07-1cc9a3f74b2b
# MAGIC keyctl show
# MAGIC ./azcopy logout

# COMMAND ----------

# MAGIC %sh
# MAGIC cat /root/.azcopy/812c8f74-a5f5-5f4e-53e8-47c7b4f0720d.log

# COMMAND ----------

df.head()

# COMMAND ----------

df[df.CURDISCOUNTCARDID == '13009598']

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

