# Databricks notebook source
import urllib.request
import os
import warnings
import sys
import numpy as np
#import findspark
#findspark.init()
from pyspark.sql.types import * 
from pyspark.sql.functions import col, lit
from pyspark.sql.functions import udf
import matplotlib
import matplotlib.pyplot as plt
import mlflow
import mlflow.spark
from pyspark.ml.feature import Imputer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml import Pipeline

from pyspark.shell import spark

from pyspark.sql import SparkSession

#from pyspark.context import SparkContext
#from pyspark.sql.session import SparkSession

#sc = SparkContext.getOrCreate()
#spark = SparkSession(sc)

# COMMAND ----------

spark = SparkSession.builder.master("local[*]").appName("local-1671554812812") .getOrCreate()

# COMMAND ----------

dataset = spark.table("hive_metastore.default.nyc_taxi_1")
display(dataset)

# COMMAND ----------

from pyspark.ml.feature import Imputer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml import Pipeline

def get_sin_cosine(value, max_value):
  sine =  np.sin(value * (2.*np.pi/max_value))
  cosine = np.cos(value * (2.*np.pi/max_value))
  return (sine.tolist(), cosine.tolist())

schema = StructType([
    StructField("sine", DoubleType(), False),
    StructField("cosine", DoubleType(), False)
])

get_sin_cosineUDF = udf(get_sin_cosine, schema)

dataset = dataset.withColumn("udfResult", get_sin_cosineUDF(col("hour_of_day"), lit(24))).withColumn("hour_sine", col("udfResult.sine")).withColumn("hour_cosine", col("udfResult.cosine")).drop("udfResult").drop("hour_of_day")

dataset = dataset.filter(dataset.totalAmount.isNotNull())

dataset = dataset.withColumn("isPaidTimeOff", col("isPaidTimeOff").cast("integer"))

numerical_cols = ["passengerCount", "tripDistance", "snowDepth", "precipTime", "precipDepth", "temperature", "hour_sine", "hour_cosine"]
categorical_cols = ["day_of_week", "month_num", "normalizeHolidayName", "isPaidTimeOff"]
label_column = "totalAmount"

stages = []

inputCols = ["passengerCount"]
outputCols = ["passengerCount"]
imputer = Imputer(strategy="median", inputCols=inputCols, outputCols=outputCols)
stages += [imputer]

assembler = VectorAssembler().setInputCols(numerical_cols).setOutputCol('numerical_features')
scaler = MinMaxScaler(inputCol=assembler.getOutputCol(), outputCol="scaled_numerical_features")
stages += [assembler, scaler]

for categorical_col in categorical_cols:
    # Category Indexing with StringIndexer
    stringIndexer = StringIndexer(inputCol=categorical_col, outputCol=categorical_col + "_index", handleInvalid="skip")
    encoder = OneHotEncoder(inputCols=[stringIndexer.getOutputCol()], outputCols=[categorical_col + "_classVector"])
    # Add stages.  These are not run here, but will run all at once later on.
    stages += [stringIndexer, encoder]
    
assemblerInputs = [c + "_classVector" for c in categorical_cols] + ["scaled_numerical_features"]
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [assembler]

partialPipeline = Pipeline().setStages(stages)
pipelineModel = partialPipeline.fit(dataset)
preppedDataDF = pipelineModel.transform(dataset)

(trainingData, testData) = preppedDataDF.randomSplit([0.7, 0.3], seed=97)


print('Created datasets')

# COMMAND ----------

def plot_regression_quality(predictions):
  p_df = predictions.select(["totalAmount",  "prediction"]).toPandas()
  true_value = p_df.totalAmount
  predicted_value = p_df.prediction

  fig = plt.figure(figsize=(10,10))
  plt.scatter(true_value, predicted_value, c='crimson')
  plt.yscale('log')
  plt.xscale('log')

  p1 = max(max(predicted_value), max(true_value))
  p2 = min(min(predicted_value), min(true_value))
  plt.plot([p1, p2], [p1, p2], 'b-')
  plt.xlabel('True Values', fontsize=15)
  plt.ylabel('Predictions', fontsize=15)
  plt.axis('equal')
  
  global image

  image = fig
  fig.savefig("LinearRegressionPrediction.png")
  plt.close(fig)
  return image

print('Created regression quality plot function')

# COMMAND ----------

from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

def train_nyc_taxi(train_data, test_data, label_column, features_column, elastic_net_param, reg_param, max_iter):
  # Evaluate metrics
  def eval_metrics(predictions):
      evaluator = RegressionEvaluator(
          labelCol=label_column, predictionCol="prediction", metricName="rmse")
      rmse = evaluator.evaluate(predictions)
      evaluator = RegressionEvaluator(
          labelCol=label_column, predictionCol="prediction", metricName="mae")
      mae = evaluator.evaluate(predictions)
      evaluator = RegressionEvaluator(
          labelCol=label_column, predictionCol="prediction", metricName="r2")
      r2 = evaluator.evaluate(predictions)
      return rmse, mae, r2

  # Start an MLflow run; the "with" keyword ensures we'll close the run even if this cell crashes
  with mlflow.start_run():
    lr = LinearRegression(featuresCol="features", labelCol=label_column, elasticNetParam=elastic_net_param, regParam=reg_param, maxIter=max_iter)
    lrModel = lr.fit(train_data)
    predictions = lrModel.transform(test_data)
    (rmse, mae, r2) = eval_metrics(predictions)

    # Print out model metrics
    print("Linear regression model (elasticNetParam=%f, regParam=%f, maxIter=%f):" % (elastic_net_param, reg_param, max_iter))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

    # Log hyperparameters for mlflow UI
    mlflow.log_param("elastic_net_param", elastic_net_param)
    mlflow.log_param("reg_param", reg_param)
    mlflow.log_param("max_iter", max_iter)
    # Log evaluation metrics
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)
    # Log the model itself
    mlflow.spark.log_model(lrModel, "model")
    modelpath = "/dbfs/mlflow/taxi_total_amount/model-%f-%f-%f" % (elastic_net_param, reg_param, max_iter)
    mlflow.spark.save_model(lrModel, modelpath)
    
    # Generate a plot
    image = plot_regression_quality(predictions)
    
    # Log artifacts (in this case, the regression quality image)
    mlflow.log_artifact("LinearRegressionPrediction.png")
    
print('Created training and evaluation method')

# COMMAND ----------

# L1 penalty, regularization parameter 0.3, 50 iterations
train_nyc_taxi(trainingData, testData, label_column, "features", 1.0, 0.3, 50)

# COMMAND ----------

from mlflow.tracking import MlflowClient
import time
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus

client = MlflowClient()

user_name = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
experiment_name = "/Repos/srai5@statestreet.com/Databricks/databricks-notebooks/Train_eval_register"

display(experiment_name)





# COMMAND ----------

experiment = client.get_experiment_by_name(experiment_name)

print(experiment)


# COMMAND ----------

experiment_id = experiment.experiment_id
runs_df = client.search_runs(experiment_id, order_by=["attributes.start_time desc"], max_results=1)
run_id = runs_df[0].info.run_uuid

model_name = "NYC Taxi Amount API"

artifact_path = "model"
model_uri = "runs:/{run_id}/{artifact_path}".format(run_id=run_id, artifact_path=artifact_path)
model_uri

# COMMAND ----------

model_details = mlflow.register_model(model_uri=model_uri, name=model_name)

# Wait until the model is ready
def wait_until_ready(model_name, model_version):
  client = MlflowClient()
  for _ in range(10):
    model_version_details = client.get_model_version(
      name=model_name,
      version=model_version,
    )
    status = ModelVersionStatus.from_string(model_version_details.status)
    print("Model status: %s" % ModelVersionStatus.to_string(status))
    if status == ModelVersionStatus.READY:
      break
    time.sleep(1)

wait_until_ready(model_details.name, model_details.version)
client = MlflowClient()

# COMMAND ----------

client.update_registered_model(
  name=model_details.name,
  description="This model forecasts the amount a taxi cab ride might cost in New York City."
)

client.update_model_version(
  name=model_details.name,
  version=model_details.version,
  description="This model version was built using Spark ML's linear regression algorithm."
)

# COMMAND ----------



# COMMAND ----------


