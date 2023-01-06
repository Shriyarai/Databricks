# Databricks notebook source
import mlflow.pyfunc
 
model_name = "NYC_Taxi_Amount_API"
 
model_version_uri = "models:/{model_name}/1".format(model_name=model_name)
 
print("Loading registered model version from URI: '{model_uri}'".format(model_uri=model_version_uri))
if mlflow.pyfunc.load_model(model_version_uri):
    print("Model Exists! Test Passed!")
else:
    print("Model does not exist.")
 
