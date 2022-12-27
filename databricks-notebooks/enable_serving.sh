curl -X POST -u token:$DATABRICKS_TOKEN https://adb-5104278588755913.13.azuredatabricks.net/api/2.0/preview/mlflow/endpoints-v2/enable \
-d '{
    "registered_model_name": "NYC Taxi Amount API 1"
}'