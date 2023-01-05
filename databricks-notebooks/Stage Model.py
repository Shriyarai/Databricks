# Databricks notebook source
import mlflow
import mlflow.spark

# COMMAND ----------

#adding comment
from mlflow.tracking import MlflowClient
import time
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus

client = MlflowClient()

# COMMAND ----------

model_name = "NYC Taxi Amount API Demo"

model_version_infos = client.search_model_versions("name = '%s'" % model_name)
print(model_version_infos)
new_model_version = max([model_version_info.version for model_version_info in model_version_infos])

# COMMAND ----------

client.transition_model_version_stage(
  name=model_name,
  version=new_model_version,
  stage="Staging",
)

model_version_details = client.get_model_version(
  name=model_name,
  version=new_model_version,
)
print("The current model stage is: '{stage}'".format(stage=model_version_details.current_stage))

# COMMAND ----------



# COMMAND ----------


