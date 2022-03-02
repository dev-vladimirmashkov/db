# Databricks notebook source
# MAGIC %sh
# MAGIC wget -O azcopy.tar.gz https://aka.ms/downloadazcopy-v10-linux
# MAGIC tar -xf azcopy.tar.gz
# MAGIC cp "$(dirname "$(find . -path ./azcopy_linux\* -type f| tail -1)")"/azcopy azcopy
# MAGIC mkdir data

# COMMAND ----------

# MAGIC %sh
# MAGIC sudo apt install keyutils

# COMMAND ----------

# MAGIC %sh
# MAGIC keyctl session workaroundSession
# MAGIC export AZCOPY_SPA_CLIENT_SECRET="@g32sDu[b1:k=FgRAApHenG24DqlYNSw"
# MAGIC ./azcopy login --service-principal --application-id 9bb009df-f8d0-472b-aadd-cda19252bd08 --tenant-id d986b6e6-6050-49f7-9e07-1cc9a3f74b2b
# MAGIC keyctl show
# MAGIC ./azcopy logout

# COMMAND ----------

# MAGIC %sh
# MAGIC git clone https://github.com/Distrotech/keyutils.git
# MAGIC cd keyutils 
# MAGIC make
# MAGIC make install

# COMMAND ----------

# MAGIC %sh
# MAGIC cd keyutils 
# MAGIC ls -lah

# COMMAND ----------

# MAGIC %sh
# MAGIC keyutils/keyctl session work77

# COMMAND ----------

# MAGIC %sh
# MAGIC cat /root/.azcopy/812c8f74-a5f5-5f4e-53e8-47c7b4f0720d.log

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

