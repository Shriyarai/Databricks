# Databricks notebook source
# MAGIC %md # Managing Models
# MAGIC 
# MAGIC There are two methods to manage models with Azure Databricks:  using the user interface or programmatically.  In the next two exercises, you will look at each technique.
# MAGIC 
# MAGIC ## Start Your Cluster
# MAGIC To get started, first attach a Databricks cluster to this notebook.  If you have not created a cluster yet, use the **Clusters** menu on the left-hand sidebar to create a new Databricks cluster.  Then, return to this notebook and attach the newly-created cluster to this notebook.
# MAGIC 
# MAGIC ## Managing a Model via the User Interface
# MAGIC 
# MAGIC In this exercise, you will once more train a model based on the `nyc-taxi` dataset.  From there, you will register the model using the Databricks user interface.  
# MAGIC 
# MAGIC The first step is to load the libraries you will use and featurize the NYC Taxi & Limousine Commission - green taxi trip records dataset.  Because you have reviewed this code in the prior notebook, explanations here will be brief until you have run the trained model.

# COMMAND ----------

import urllib.request
import os
import warnings
import sys
import numpy as np
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

dataset = spark.sql("select * from nyc_taxi_1")

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

print('Data preparation work completed.')

# COMMAND ----------

# MAGIC %md With this data in place, create a function to plot the quality of the regression model based on predicted amounts versus actual amounts.

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

# MAGIC %md The following method trains the regression model and uses MLflow Tracking to record parameters, metrics, model, and a plot which compares actual versus predicted amounts spent on taxi rides.  This is essentially the same model as what you used in the prior lesson, although there is a minor change in lines 40-43, which you will take advantage of in the next exercise.

# COMMAND ----------

from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

def train_nyc_taxi(train_data, test_data, label_column, features_column, elastic_net_param, reg_param, max_iter, model_name=None):
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
    if model_name is None:
      mlflow.spark.log_model(lrModel, "model")
    else:
      mlflow.spark.log_model(lrModel, artifact_path="model", registered_model_name=model_name)
    modelpath = "/dbfs/mlflow/taxi_total_amount_2/model-%f-%f-%f" % (elastic_net_param, reg_param, max_iter)
    mlflow.spark.save_model(lrModel, modelpath)
    
    # Generate a plot
    image = plot_regression_quality(predictions)
    
    # Log artifacts (in this case, the regression quality image)
    mlflow.log_artifact("LinearRegressionPrediction.png")
    
print('Created training and evaluation method')

# COMMAND ----------

# MAGIC %md Remove any prior executions of this script.  Note that the folder is now `dbfs:/mlflow/taxi_total_amount_2` instead of `taxi_total_amount`.  This way, you will not overwrite executions from the prior lab.

# COMMAND ----------

# MAGIC %fs rm -r dbfs:/mlflow/taxi_total_amount_2

# COMMAND ----------

# MAGIC %md Train the model with what were the most successful hyperparameters in the prior lab.

# COMMAND ----------

# L1 penalty, regularization parameter 0.3, 50 iterations
train_nyc_taxi(trainingData, testData, label_column, "features", 1.0, 0.3, 50)

# COMMAND ----------

# MAGIC %md ### Registering the Model
# MAGIC 
# MAGIC Select the **Experiment** option in the notebook context bar to display the Experiment sidebar.  In this sidebar, select the `spark` Link for your experiment run.  This will open the experiment run's details in a new browser tab and navigate to the model itself.
# MAGIC 
# MAGIC On the model page, select **Register Model** to register a new model.  In the **Model** drop-down list, select **+ Create New Model** and enter the name **NYC Taxi Amount UI**.  Then, select **Register**.  Registration may take a couple of minutes to complete.  You may need to refresh the tab to change the model registration status changes from **Registration pending...** to its **Registered** status.
# MAGIC 
# MAGIC ### Serving the Model
# MAGIC 
# MAGIC From here, navigate to the **Models** page using the menu on the left-hand side.  You will see the `NYC Taxi Amount UI` model.  Select the model link to view details about the model.  Note that you can add tags the model or view different versions of a model.  To activate the model, select the **Serving** tab and then select **Enable Serving**.  This will set up a single-node cluster intended for generating predictions.  This process may take several minutes, so be patient.  You may need to refresh your browser occasionally to see updates.
# MAGIC 
# MAGIC After the registration status changes to **Ready** from **Pending**, you can generate a prediction through your browser.  One way to test this is to select the **Browser** button in the **Call The Model** section and enter a JSON array into the **Request** field.  This particular model, however, is fairly complex, so it's actually easier to call it from code.  We will do that in the next exercise.
# MAGIC 
# MAGIC ### Deleting the Model
# MAGIC 
# MAGIC Once you are done testing the model, select the drop-down symbol next to **Registered Models > NYC Taxi Amount UI** in the header section and then choose **Delete**.  Confirm that you wish to delete the model.  It will stop serving the current model and delete the model from the registry.

# COMMAND ----------

# MAGIC %md ## Managing a Model via Code
# MAGIC 
# MAGIC In addition to the user interface, it is possible to manage models via code.  In this exercise, you will take the same trained model as in the prior exercise and manage the model using the `MlflowClient` library in Python.

# COMMAND ----------

from mlflow.tracking import MlflowClient
import time
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus

client = MlflowClient()

# COMMAND ----------

# MAGIC %md ### Retrieve the Model
# MAGIC 
# MAGIC The first step will be to retrieve the model you created in the prior exercise.  To do this, first retrieve the experiment that you created in the prior exercise.  Because you did not specify an experiment name, the name will be the same as this notebook's name.

# COMMAND ----------

user_name = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
experiment_name = "/Users/{user_name}/03 - Managing Experiments and Models/02 - Managing Models".format(user_name=user_name)

experiment = client.get_experiment_by_name(experiment_name)

# COMMAND ----------

# MAGIC %md Next, retrieve the latest run of model training.  This is located in a folder named by the run's unique identifier (`run_uuid`).  From there, you wrote the model to a `model` folder in `train_nyc_taxi()`.

# COMMAND ----------

experiment_id = experiment.experiment_id
runs_df = client.search_runs(experiment_id, order_by=["attributes.start_time desc"], max_results=1)
run_id = runs_df[0].info.run_uuid

model_name = "NYC Taxi Amount API"

artifact_path = "model"
model_uri = "runs:/{run_id}/{artifact_path}".format(run_id=run_id, artifact_path=artifact_path)
model_uri

# COMMAND ----------

# MAGIC %md ### Register Model
# MAGIC 
# MAGIC The next step is to register the model.  This model will be registered under the name `NYC Taxi Amount API`.  Once the cell returns "Model status: READY", the model will be available.  This may take a few minutes.

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

# COMMAND ----------

# MAGIC %md Once the model is available, you can update the currently registered model.  The following method calls update the model description and the model version's description, respectively.
# MAGIC 
# MAGIC Each model has one or more versions, which represent iterations on the trained model.  Creating descriptions for these model versions can help you keep track of changes over time, such as using a new algorithm.

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

# MAGIC %md ### Model Staging
# MAGIC 
# MAGIC MLflow allows multiple versions of a model to exist at the same time.  To remove ambiguity in which model should be in use at any time, you can stage models, using states such as `Staging` or `Production`.
# MAGIC 
# MAGIC Use the `Production` stage on the version of the model you want to use for inference.  The process to do this follows.

# COMMAND ----------

client.transition_model_version_stage(
  name=model_details.name,
  version=model_details.version,
  stage='Production',
)
model_version_details = client.get_model_version(
  name=model_details.name,
  version=model_details.version,
)
print("The current model stage is: '{stage}'".format(stage=model_version_details.current_stage))

latest_version_info = client.get_latest_versions(model_name, stages=["Production"])
latest_production_version = latest_version_info[0].version
print("The latest production version of the model '%s' is '%s'." % (model_name, latest_production_version))

# COMMAND ----------

# MAGIC %md ### Model Inference
# MAGIC 
# MAGIC The following function will allow you to forecast the cost of a taxi ride in New York City given certain conditions.

# COMMAND ----------

import mlflow.pyfunc

def forecast_nyc_taxi_amount(model_name, model_stage, df):
  model_uri = "models:/{model_name}/{model_stage}".format(model_name=model_name,model_stage=model_stage)
  print("Loading registered model version from URI: '{model_uri}'".format(model_uri=model_uri))
  model = mlflow.pyfunc.load_model(model_uri)
  return model.predict(df)

# COMMAND ----------

# MAGIC %md With this function in place, build a sample input and generate the forecast for the `Production` model.  Use the `testData` DataFrame that you created earlier in this lab, as it has all of the inputs in the right shape for performing inference.

# COMMAND ----------

model_stage = "Production"
df = testData.head(1)
forecast_nyc_taxi_amount(model_name, model_stage, df)

# COMMAND ----------

# MAGIC %md ### Model Versioning
# MAGIC 
# MAGIC Creating a new version of a model is easy.  In this case, run the `train_nyc_taxi()` method and specify a new parameter which defines the model name.  This will write a new version of the current model while retaining the current `Production` version.

# COMMAND ----------

# Create a new version
# L2 penalty, regularization parameter 0.3, 500 iterations
train_nyc_taxi(trainingData, testData, label_column, "features", 0.0, 0.3, 500, model_name)

# COMMAND ----------

# MAGIC %md Now, retrieve the latest version of the `NYC Taxi Amount API` model.

# COMMAND ----------

model_version_infos = client.search_model_versions("name = '%s'" % model_name)
new_model_version = max([model_version_info.version for model_version_info in model_version_infos])

wait_until_ready(model_name, new_model_version)

# COMMAND ----------

# MAGIC %md Use the model version description to explain how this model differs from the others.  In this case, you changed the value of the *max_iter* parameter from 50 to 500 and also changed the ElasticNet parameter.

# COMMAND ----------

client.update_model_version(
  name=model_name,
  version=new_model_version,
  description="This model version has changed the max number of iterations to 500 and minimizes L2 penalties."
)

# COMMAND ----------

# MAGIC %md Before moving this model to production, you can stage the model by moving this version to `Staging`.

# COMMAND ----------

client.transition_model_version_stage(
  name=model_name,
  version=new_model_version,
  stage="Staging",
)

# COMMAND ----------

# MAGIC %md The reason the `forecast_nyc_taxi_amount()` function included a model stage is to allow testing of the `Staging` model before transitioning it to `Production`.
# MAGIC 
# MAGIC Note that the predicted amount is slightly different from the model in production.

# COMMAND ----------

# Generate a prediction for the new model
forecast_nyc_taxi_amount(model_name, "Staging", df)

# COMMAND ----------

# MAGIC %md It looks like this didn't change the results very much, but there is a small difference.  Let's say that you are confident in the new model and are ready to make it the new production model.

# COMMAND ----------

# MAGIC %md ### Transitioning a New Version of a Model
# MAGIC 
# MAGIC Now that the `Staging` model version is out, the next step is to transition the latest model version to `Production`.  Do this using the same `transition_model_version_stage()` method as before.

# COMMAND ----------

client.transition_model_version_stage(
  name=model_name,
  version=new_model_version,
  stage="Production",
)

# COMMAND ----------

client.search_model_versions("name = '%s'" % model_name)

# COMMAND ----------

# MAGIC %md Now both model versions are tagged as production.  Which one will Azure Databricks use?

# COMMAND ----------

forecast_nyc_taxi_amount(model_name, "Production", df)

# COMMAND ----------

# MAGIC %md It turns out that Azure Databricks looks for the latest model version with a given tag.  We can tell because the predicted amount is the same amount we saw from the most recently trained model--in other words, the one we most recently promoted to `Production`.
# MAGIC 
# MAGIC This means that you could conceivably have several `Production` versions of models running concurrently.  But a more practical plan is to archive the old model.
# MAGIC 
# MAGIC ### Archiving a Model Version
# MAGIC 
# MAGIC In order to archive a model version, call `transition_model_version_stage()` once more, but use the `Archived` stage.

# COMMAND ----------

client.transition_model_version_stage(
  name=model_name,
  version=model_details.version,
  stage="Archived",
)

# COMMAND ----------

# MAGIC %md If you wish to go further and delete a model version, a method is available for that as well.

# COMMAND ----------

client.delete_model_version(
   name=model_name,
   version=model_details.version,
)

# COMMAND ----------

# MAGIC %md Before you are able to delete a model, you must transition all `Production` or `Staging` versions to `Archived`.  This is a safety precaution to prevent accidentally deleting a model being served in production.  The following cells will transition the new model version to `Archived` and then delete this model version.

# COMMAND ----------

# Need to transition before deleting
client.transition_model_version_stage(
  name=model_name,
  version=new_model_version,
  stage="Archived",
)

# COMMAND ----------

client.delete_model_version(
   name=model_name,
   version=new_model_version
)

# COMMAND ----------

# MAGIC %md Finally, you will be able to delete the registered model.

# COMMAND ----------

client.delete_registered_model(name=model_name)
