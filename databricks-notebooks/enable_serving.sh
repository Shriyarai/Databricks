curl -X POST -u token:$DATABRICKS_TOKEN https://dbc-1f875879-7090.cloud.databricks.com/api/2.0/preview/mlflow/endpoints-v2/enable \
-d '{
    "registered_model_name": "NYC Taxi Amount API GA"
}'