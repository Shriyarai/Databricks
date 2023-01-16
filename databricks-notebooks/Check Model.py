# Databricks notebook source
# MAGIC %md ### Model Inference
# MAGIC 
# MAGIC The following function will allow you to forecast the cost of a taxi ride in New York City given certain conditions.

# COMMAND ----------

dataset = spark.sql("SELECT * FROM `hive_metastore`.`default`.`nyc_taxi_traintest`")

(trainingData, testData) = dataset.randomSplit([0.7, 0.3], seed=97)

# print(testData.head(1))


# COMMAND ----------

import mlflow.pyfunc

def forecast_nyc_taxi_amount(model_name, model_stage, df):
  model_uri = "models:/{model_name}/{model_stage}".format(model_name=model_name,model_stage=model_stage)
  print("Loading registered model version from URI: '{model_uri}'".format(model_uri=model_uri))
  model = mlflow.pyfunc.load_model(model_uri)
  return model.predict(df)

# COMMAND ----------

model_name = "NYC_Taxi_Amount_API"
model_stage = "Production"
df = testData.head(1)
# print(df)
forecast_nyc_taxi_amount(model_name, model_stage, df)

# COMMAND ----------


