# Databricks notebook source
import mlflow
import mlflow.spark

# COMMAND ----------

from mlflow.tracking import MlflowClient
import time
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus

client = MlflowClient()

# COMMAND ----------

model_name = "NYC_Taxi_Amount_API"

model_version_infos = client.search_model_versions("name = '%s'" % model_name)
print(model_version_infos)
new_model_version = max([model_version_info.version for model_version_info in model_version_infos])

# COMMAND ----------

for model_version_info in model_version_infos:
    if model_version_info.version == new_model_version and model_version_info.current_stage == "Staging":
        client.transition_model_version_stage(
            name=model_name,
            version=new_model_version,
            stage='Production',
            )
        
model_version_details = client.get_model_version(
  name=model_name,
  version=new_model_version,
)
print("The current model stage is: '{stage}'".format(stage=model_version_details.current_stage))

latest_version_info = client.get_latest_versions(model_name, stages=["Production"])
latest_production_version = latest_version_info[0].version
print("The latest production version of the model '%s' is '%s'." % (model_name, latest_production_version))

# COMMAND ----------

# model_version_infos = client.search_model_versions("name = '%s'" % model_name)

for model_version_info in model_version_infos:
    if model_version_info.version != latest_production_version and model_version_info.current_stage != "Archived":
        client.transition_model_version_stage(
            name=model_name,
            version=model_version_info.version,
            stage="Archived",
        )

# COMMAND ----------


