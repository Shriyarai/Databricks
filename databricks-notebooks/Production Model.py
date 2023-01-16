# Databricks notebook source
# MAGIC %md ### Transitioning a New Version of a Model
# MAGIC 
# MAGIC Now that the `Staging` model version is out, the next step is to compare the latest model version to `Production` and transition it accordingly.  Do this using the same `transition_model_version_stage()` method as before.

# COMMAND ----------

import mlflow
import mlflow.spark

# COMMAND ----------

from mlflow.tracking import MlflowClient
import time
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus

client = MlflowClient()

# COMMAND ----------

model_name = "NYC_Taxi_Amount_API"
model_stage = "Production"

model_uri = "models:/{model_name}/{model_stage}".format(model_name=model_name,model_stage=model_stage)
print("Loading registered model version from URI: '{model_uri}'".format(model_uri=model_uri))
model = mlflow.pyfunc.load_model(model_uri)
print(model)

# COMMAND ----------

model_name = "NYC_Taxi_Amount_API"

prod_model_info = client.get_latest_versions(model_name, stages=["Production"])
# print(prod_model_info)
# print(prod_model_info[0].run_id)
prod_run = mlflow.get_run(run_id=prod_model_info[0].run_id)

print(prod_run.data.metrics['r2'])

prod_run_r2 = prod_run.data.metrics['r2']

# COMMAND ----------

print(client.get_metric_history(run_id=prod_model_info[0].run_id, key='rmse'))

# COMMAND ----------

model_name = "NYC_Taxi_Amount_API"

model_version_infos = client.search_model_versions("name = '%s'" % model_name)
print(model_version_infos)
new_model_version = max([model_version_info.version for model_version_info in model_version_infos])
print(new_model_version)

# COMMAND ----------

for model_version_info in model_version_infos:
    if model_version_info.version == new_model_version and model_version_info.current_stage == "Staging":
        latest_run = mlflow.get_run(run_id=model_version_info.run_id)
        latest_run_r2 = latest_run.data.metrics['r2']
        print(latest_run_r2)

# COMMAND ----------

if latest_run_r2 > prod_run_r2:
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

# MAGIC %md
# MAGIC ### Archiving a Model Version
# MAGIC 
# MAGIC In order to archive a model version, call `transition_model_version_stage()` once more, but use the `Archived` stage.

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

# MAGIC %md Before you are able to delete a model, you must transition all `Production` or `Staging` versions to `Archived`.  This is a safety precaution to prevent accidentally deleting a model being served in production.  
