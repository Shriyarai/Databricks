# Databricks notebook source
import time
import json
import os
from databricks_cli.sdk.api_client import ApiClient
# for working with jobs
from databricks_cli.jobs.api import JobsApi
# for working with job runs
from databricks_cli.runs.api import RunsApi

# --------- Setup authentication with Databricks API ------------
# https://docs.databricks.com/dev-tools/python-api.html
# adding comment 

api_client = ApiClient(
    host=os.getenv("DATABRICKS_HOST"),
    token=os.getenv("DATABRICKS_TOKEN")
)


# COMMAND ----------



jobs_api = JobsApi(api_client)  # https://github.com/databricks/databricks-cli/blob/main/databricks_cli/jobs/api.py
runs_api = RunsApi(api_client)  # https://github.com/databricks/databricks-cli/blob/main/databricks_cli/runs/api.py

# --------------- TRIGGER TRAINING AND EVALUATION -----------------
train_eval_run = jobs_api.run_now(73317636492511,
                                  jar_params=None,
                                  notebook_params=None,
                                  python_params=None,
                                  spark_submit_params=None)

# ---------------- WAIT FOR RUN TO FINISH -------------------------
# GET /2.1/jobs/runs/get
# Must provide "run_id"
status = runs_api.get_run(run_id=train_eval_run["run_id"])
print(status)
while status["state"]["life_cycle_state"] != "TERMINATED":
    # print("waiting")
    time.sleep(30)
    status = runs_api.get_run(run_id=train_eval_run["run_id"])
print(status)

# with open("run_info.json","w") as f:
#     data = json.dumps({
#         "run_id": train_eval_run["run_id"]
#     })
#     f.write(data)

# --------------- EXPORTING THE RUN_ID RATHER THAN WRITING TO A FILE --------------

# os.environ['RUN_ID'] = str(train_eval_run["run_id"])
# print(train_eval_run["run_id"])

# COMMAND ----------

display("Ran successfully")
