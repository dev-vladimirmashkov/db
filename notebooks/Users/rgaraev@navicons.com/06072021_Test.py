# Databricks notebook source
# MAGIC %sh
# MAGIC wget -O azcopy.tar.gz https://aka.ms/downloadazcopy-v10-linux
# MAGIC tar -xf azcopy.tar.gz
# MAGIC cp "$(dirname "$(find . -path ./azcopy_linux\* -type f| tail -1)")"/azcopy azcopy
# MAGIC mkdir data
# MAGIC mkdir model

# COMMAND ----------

import numpy as np
import pandas as pd
df = pd.DataFrame()
df["a"]=0
l=[]
# просто делает какие-то вычисления :)
for i in range(3000):
  #print(np.exp(i)/np.exp(i+np.random.randint(1,10)))
  l.append(np.exp(i)/np.exp(i+np.random.randint(1,10)))
df["a"] = l
df.to_parquet('data/06072021_test.parquet', index=False)

# COMMAND ----------

# MAGIC %sh
# MAGIC 
# MAGIC ./azcopy copy data/06072021_test.parquet "https://alkordatalake.blob.core.windows.net/letoile/raw/ml/forecast.parquet?sv=2020-04-08&ss=btqf&srt=sco&st=2021-06-01T12%3A00%3A00Z&se=2023-12-31T12%3A00%3A00Z&sp=rwlac&sig=LDuVmtTYZ4r71Xqe1moPRosscCRqpkHbSEPByJTyT6M%3D"