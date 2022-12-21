# Databricks notebook source
# MAGIC %md
# MAGIC # Getting Started with Azure Databricks
# MAGIC 
# MAGIC **Technical Accomplishments:**
# MAGIC - Set the stage for learning on the Databricks platform
# MAGIC - Create the cluster
# MAGIC - Discover the workspace, import table data
# MAGIC - Demonstrate how to develop & execute code within a notebook
# MAGIC - Review the various "Magic Commands"
# MAGIC - Introduce the Databricks File System (DBFS)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Attach notebook to your cluster
# MAGIC Before executing any cells in the notebook, you need to attach it to your cluster. Make sure that the cluster is running.
# MAGIC 
# MAGIC In the notebook's toolbar, select the drop down arrow next to Detached, and then select your cluster under Attach to.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Working with notebooks
# MAGIC 
# MAGIC A notebook is a web-based interface to a document that contains 
# MAGIC * runnable code
# MAGIC * visualizations
# MAGIC * descriptive text
# MAGIC 
# MAGIC To create a notebook, click on `Workspace`, browse into the desired folder, right click and choose `Create` then select `Notebook`.

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC A notebook contains multiple cells. Each cell has a specific type. 
# MAGIC 
# MAGIC A default programming language is configured when creating the notebook and it will be implicitly used for new cells.
# MAGIC 
# MAGIC #### Magic commands
# MAGIC 
# MAGIC We can override the cell's default programming language by using one of the following *magic commands* at the start of the cell:
# MAGIC 
# MAGIC * `%python` for cells running python code
# MAGIC * `%scala` for cells running scala code
# MAGIC * `%r` for cells running R code
# MAGIC * `%sql` for cells running sql code
# MAGIC   
# MAGIC Additional magic commands are available:
# MAGIC 
# MAGIC * `%md` for descriptive cells using markdown
# MAGIC * `%sh` for cells running shell commands
# MAGIC * `%run` for cells running code defined in a separate notebook
# MAGIC * `%fs` for cells running code that uses `dbutils` commands
# MAGIC   

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC To run a cell use one of the following options:
# MAGIC   * **CTRL+ENTER** or **CMD+RETURN**
# MAGIC   * **SHIFT+ENTER** or **SHIFT+RETURN** to run the cell and move to the next one
# MAGIC   * Using **Run Cell**, **Run All Above** or **Run All Below** as seen here<br/><img style="box-shadow: 5px 5px 5px 0px rgba(0,0,0,0.25); border: 1px solid rgba(0,0,0,0.25);" src="https://files.training.databricks.com/images/notebook-cell-run-cmd.png"/>

# COMMAND ----------

#assuming the default language for a notebook was set to Python
print("I'm running Python!")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Below we use a simple python function to convert Celsius degrees to Fahrenheit degrees.

# COMMAND ----------

# MAGIC %python
# MAGIC 
# MAGIC #convert celsius to fahrenheit
# MAGIC def celsiusToFahrenheit(source_temp=None):
# MAGIC     return(source_temp * (9.0/5.0)) + 32.0    
# MAGIC         
# MAGIC #input values - celsius
# MAGIC a = [1, 2, 3, 4, 5]
# MAGIC print(a)
# MAGIC 
# MAGIC #convert all
# MAGIC b = map(lambda x: celsiusToFahrenheit(x), a)
# MAGIC print(list(b))

# COMMAND ----------

# MAGIC %md
# MAGIC ##![Spark Logo Tiny](https://files.training.databricks.com/images/wiki-book/general/logo_spark_tiny.png) Databricks File System - DBFS
# MAGIC 
# MAGIC We've already imported data into Databricks by uploading our files.
# MAGIC 
# MAGIC Databricks is capable of mounting external/remote datasources as well.
# MAGIC 
# MAGIC DBFS allows you to mount storage objects so that you can seamlessly access data without requiring credentials.
# MAGIC Allows you to interact with object storage using directory and file semantics instead of storage URLs.
# MAGIC Persists files to object storage, so you won’t lose data after you terminate a cluster.
# MAGIC 
# MAGIC * DBFS is a layer over a cloud-based object store
# MAGIC * Files in DBFS are persisted to the object store
# MAGIC * The lifetime of files in the DBFS are **NOT** tied to the lifetime of our cluster
# MAGIC 
# MAGIC See also <a href="https://docs.azuredatabricks.net/user-guide/dbfs-databricks-file-system.html" target="_blank">Databricks File System - DBFS</a>.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Databricks Utilities - dbutils
# MAGIC * You can access the DBFS through the Databricks Utilities class (and other file IO routines).
# MAGIC * An instance of DBUtils is already declared for us as `dbutils`.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC The `mount` command allows to use remote storage as if it were a local folder available in the Databricks workspace
# MAGIC 
# MAGIC ```
# MAGIC dbutils.fs.mount(
# MAGIC   source = f"wasbs://dev@{data_storage_account_name}.blob.core.windows.net",
# MAGIC   mount_point = data_mount_point,
# MAGIC   extra_configs = {f"fs.azure.account.key.{data_storage_account_name}.blob.core.windows.net": data_storage_account_key})
# MAGIC ```

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC To show available DBFS mounts:

# COMMAND ----------

# MAGIC %fs 
# MAGIC mounts

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC To show available tables:

# COMMAND ----------

# MAGIC %fs
# MAGIC ls /FileStore/tables

# COMMAND ----------

# MAGIC %md
# MAGIC Additional help is available via `dbutils.help()` and for each sub-utility: `dbutils.fs.help()`, `dbutils.meta.help()`, `dbutils.notebook.help()`, `dbutils.widgets.help()`.
# MAGIC 
# MAGIC See also <a href="https://docs.azuredatabricks.net/user-guide/dbutils.html" target="_blank">Databricks Utilities - dbutils</a>

# COMMAND ----------

dbutils.fs.help()
