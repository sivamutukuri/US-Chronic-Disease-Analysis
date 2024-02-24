# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC ## Overview
# MAGIC
# MAGIC This notebook will show you how to create and query a table or DataFrame that you uploaded to DBFS. [DBFS](https://docs.databricks.com/user-guide/dbfs-databricks-file-system.html) is a Databricks File System that allows you to store data for querying inside of Databricks. This notebook assumes that you have a file already inside of DBFS that you would like to read from.
# MAGIC
# MAGIC This notebook is written in **Python** so the default cell type is Python. However, you can use different languages by using the `%LANGUAGE` syntax. Python, Scala, SQL, and R are all supported.

# COMMAND ----------

# File location and type
file_location = "/FileStore/tables/U_S__Chronic_Disease_Indicators__CDI_.csv"
file_type = "csv"

# CSV options
infer_schema = "false"
first_row_is_header = "true"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

display(df)

# COMMAND ----------

# Create a view or table

temp_table_name = "U_S__Chronic_Disease_Indicators__CDI__csv"

df.createOrReplaceTempView(temp_table_name)

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC /* Query the created temp table in a SQL cell */
# MAGIC
# MAGIC select * from `U_S__Chronic_Disease_Indicators__CDI__csv`

# COMMAND ----------

# With this registered as a temp view, it will only be available to this particular notebook. If you'd like other users to be able to query this table, you can also create a table from the DataFrame.
# Once saved, this table will persist across cluster restarts as well as allow various users across different notebooks to query this data.
# To do so, choose your table name and uncomment the bottom line.

permanent_table_name = "U_S__Chronic_Disease_Indicators__CDI__csv"

# df.write.format("parquet").saveAsTable(permanent_table_name)

# COMMAND ----------

# MAGIC %md
# MAGIC GeoLocation: This column likely contains information related to the geographic location of the data point, possibly in the form of latitude and longitude coordinates.
# MAGIC
# MAGIC ResponseID: This column may contain an identifier for the response or data point.
# MAGIC
# MAGIC LocationID: This column may contain an identifier for the location associated with the data point.
# MAGIC
# MAGIC TopicID: This column may contain an identifier for the topic or subject area of the data point.
# MAGIC
# MAGIC QuestionID: This column may contain an identifier for the specific question or prompt associated with the data point.
# MAGIC
# MAGIC DataValueTypeID: This column may contain an identifier for the type of data being reported, such as a count, percentage, or rate.
# MAGIC
# MAGIC StratificationCategoryID1: This column may contain an identifier for the first category used to stratify the data.
# MAGIC
# MAGIC StratificationID1: This column may contain an identifier for the specific stratification used within the first category.
# MAGIC
# MAGIC StratificationCategoryID2: This column may contain an identifier for the second category used to stratify the data.
# MAGIC
# MAGIC StratificationID2: This column may contain an identifier for the specific stratification used within the second category.
# MAGIC
# MAGIC StratificationCategoryID3: This column may contain an identifier for the third category used to stratify the data.
# MAGIC
# MAGIC StratificationID3: This column may contain an identifier for the specific stratification used within the third category.
# MAGIC

# COMMAND ----------

print("Number of rows:", df.count())
print("Number of columns:", len(df.columns))

# COMMAND ----------

# MAGIC %md
# MAGIC # cleaning the data

# COMMAND ----------

from pyspark.sql.functions import col
# finding out the null values
null_counts = []
for col_name in df.columns:
    null_count = df.filter(col(col_name).isNull()).count()
    null_counts.append(null_count)


# COMMAND ----------

null_counts

# COMMAND ----------

print("Original schema:")
df.printSchema()


# COMMAND ----------

df = df.drop("StratificationID3","StratificationCategoryID3","StratificationID2","StratificationCategoryID2","ResponseID","StratificationCategory2","Stratification2","StratificationCategory3","Stratification3","Response")


df.printSchema()

# COMMAND ----------

display(df)

# COMMAND ----------

df = df.drop("DataValueFootnoteSymbol", "DatavalueFootnote", "TopicID", "QuestionID", "DataValueTypeID", "StratificationCategoryID1", "StratificationID1","LocationID")# droping the unwanted columns


# COMMAND ----------

from pyspark.sql.functions import count

display(df.select([(count(c) / count('*')).alias(c) for c in df.columns]))#percentage of non-null values for each column 


# COMMAND ----------

df = df.na.drop(subset=["DataValue"]) # droping the null values in datavalue column
display(df)

# COMMAND ----------

df = df.na.drop(subset=["DataValueUnit"])# droping the null values in DataValueUnit column
display(df)

# COMMAND ----------

from pyspark.sql.functions import count

display(df.select([(count(c) / count('*')).alias(c) for c in df.columns]))#percentage of non-null values for each column 

# COMMAND ----------

df.printSchema()


# COMMAND ----------

from pyspark.sql.functions import col
# Convert the age_str column to a numeric type
df = df.withColumn("YearStart", col("YearStart").cast("integer"))
df = df.withColumn("YearEnd", col("YearEnd").cast("integer"))
df = df.withColumn("DataValue", col("DataValue").cast("double"))
df = df.withColumn("DataValueAlt", col("DataValueAlt").cast("double"))
df = df.withColumn("LowConfidenceLimit", col("LowConfidenceLimit").cast("double"))
df = df.withColumn("HighConfidenceLimit", col("HighConfidenceLimit").cast("double"))

df.printSchema()

# COMMAND ----------

print("Number of rows:", df.count())
print("Number of columns:", len(df.columns))

# COMMAND ----------

# MAGIC %md
# MAGIC #  Data Analysis

# COMMAND ----------

# MAGIC %md
# MAGIC How has the prevalence of chronic diseases changed over time?
# MAGIC
# MAGIC

# COMMAND ----------

df = df.withColumn("DataValue", col("DataValue").cast("double"))

# Group the data by year and calculate the mean prevalence for each year
prevalence_by_year = df.groupBy("YearStart").mean("DataValue")


# COMMAND ----------

prevalence_by_year.show()

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns
sns_data = prevalence_by_year.toPandas()
plt.figure(figsize=(10,5))
sns.set_style("whitegrid")
sns.barplot(x="YearStart", y="avg(DataValue)", data=sns_data)
plt.xlabel("Year")
plt.ylabel("Mean of DataValue")
plt.title("prevalence of chronic diseases changed over time")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC What is the most common chronic diseases? 
# MAGIC

# COMMAND ----------

topic_counts = df.groupBy("Topic").count()
topic_counts = topic_counts.sort("count", ascending=False)
display(topic_counts)

# COMMAND ----------


# Convert the PySpark DataFrame to a Pandas DataFrame
topic_counts_pd = topic_counts.toPandas()

# Sort the topics by count in descending order
topic_counts_pd = topic_counts_pd.sort_values("count", ascending=False)

# Create the Seaborn barplot
sns.set(style="whitegrid")
plt.figure(figsize=(10,6))
sns.barplot(x="count", y="Topic", data=topic_counts_pd)

plt.xlabel("values")
plt.ylabel("Types")
plt.xticks(rotation=90)

plt.title("chronic diseases")
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC Which state has the highest prevalence of heart disease?

# COMMAND ----------

heart_disease = df.filter(col("Topic").like("%Cardiovascular%"))
mean_prevalence_by_state = heart_disease.groupBy("LocationDesc").mean("DataValue")
state = mean_prevalence_by_state.orderBy(col("avg(DataValue)").desc())
state = state.filter(col("LocationDesc") != "United States")
state.show()

# COMMAND ----------

pandas_df = state.toPandas()

# Create the plot using Seaborn
plt.figure(figsize=(10, 8))
sns.barplot(y="avg(DataValue)", x="LocationDesc", data=pandas_df)
plt.ylabel("values")
plt.xticks(rotation=90)
plt.title("prevalence of heart disease for each region")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Is there any difference between males and females in heart disease?

# COMMAND ----------

from pyspark.sql.functions import col

# Filter the data for heart disease
heart_disease = df.filter(col("Topic").like("%Cardiovascular%"))

# Group the data by gender and calculate the mean prevalence for each gender
mean_prevalence_by_gender = heart_disease.groupBy("Stratification1").mean("DataValue")

# Show the mean prevalences for males and females
mean_prevalence_by_gender.show()

# Calculate the difference in mean prevalences between males and females
male_prevalence = mean_prevalence_by_gender.filter(col("Stratification1") == "Male").collect()[0]["avg(DataValue)"]
female_prevalence = mean_prevalence_by_gender.filter(col("Stratification1") == "Female").collect()[0]["avg(DataValue)"]
overall = mean_prevalence_by_gender.filter(col("Stratification1") == "Overall").collect()[0]["avg(DataValue)"]
print("Mean prevalence for males: {:.2f}".format(male_prevalence))
print("Mean prevalence for females: {:.2f}".format(female_prevalence))

# COMMAND ----------

heart_disease = df.filter(col("Topic").like("%Cardiovascular%"))
group = heart_disease.groupBy("Stratification1").mean("DataValue")

# Filter to only include Male and Female rows
group = group.filter(col("Stratification1").isin(["Male", "Female"]))
pandas_df = group.toPandas()

# Create the plot using Seaborn
pandas_df.plot(kind="bar")
plt.xlabel("Gender")
plt.ylabel("AVG_DataValue")
plt.title("Difference between males and females in heart disease category")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC What is the prevalence of heart disease by gender and state in the United States? How does the prevalence vary across different states for males and females?

# COMMAND ----------

from pyspark.sql.functions import avg
from pyspark.sql.functions import desc

# Group by state and gender and calculate the mean prevalence
group_state_gender = (
    heart_disease.groupBy("LocationDesc", "Stratification1")
                 .agg(avg("DataValue").alias("mean_prevalence"))
)

# Pivot the dataframe to have state as rows and gender as columns
group_state_gender_pivot = (
    group_state_gender.groupBy("LocationDesc")
                      .pivot("Stratification1", ["Male", "Female"])
                      .agg(avg("mean_prevalence"))
                      .sort("Female", ascending=False))

# COMMAND ----------

group_state_gender_pivot.show()

# COMMAND ----------

pandas_df = group_state_gender_pivot.toPandas()
pandas_df.plot(kind="bar", figsize=(10, 8))
plt.xlabel("States")
plt.ylabel("Mean Prevalence")
plt.title("Prevalence of Heart Disease by Gender and region")
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC What is the trend in the mean percentage of individuals with diabetes in California, Texas, and New York from 2011 to 2018?

# COMMAND ----------


diabetes = df.filter(df["Topic"] == "Diabetes")
# Filter for specific states
states = ["California", "Texas", "New York"]
diabetes = diabetes.filter(diabetes["LocationDesc"].isin(states))

# Group the data by state and year, and calculate the mean prevalence of diabetes
grouped = diabetes.groupBy(["LocationDesc", "YearStart"]).agg(avg("DataValue").alias("mean_prevalence"))
pandas_df = grouped.toPandas()
# Create the plot
fig = plt.figure(figsize=(10, 8))
sns.lineplot(data=pandas_df, x="YearStart", y="mean_prevalence", hue="LocationDesc")
plt.title("Prevalence of Diabetes by States and Year")
plt.xlabel("Year")
plt.ylabel("Prevalence of Diabetes")

# Display the plot
plt.show()


# COMMAND ----------

cancer = df.filter(col("Topic").like("%Cancer%"))

# Filter for years 2010-2020
cancer = cancer.filter((col("YearStart") >= 2010) & (col("YearStart") <= 2020))

# Calculate mean incidence rate by year
incidence_by_year = cancer.groupBy("YearStart").agg(avg("DataValue").alias("mean_incidence"))
fig = plt.figure(figsize=(10, 8))
sns.lineplot(data=incidence_by_year.toPandas(), x="YearStart", y="mean_incidence")
plt.title("Trend of Cancer Incidence in the United States (2010-2020)")
plt.xlabel("Year")
plt.ylabel("Incidence Rate")

# COMMAND ----------

# MAGIC %md
# MAGIC What are the top types of cancer mortality in the dataset?
# MAGIC
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

import pyspark.sql.functions as F
cancer = df.filter(df["Topic"] == "Cancer").groupBy("Question").count()
cancer = cancer.filter(col("Question").like("%mortality%"))
cancer = cancer.withColumn("Question", F.split(F.col("Question"), ",")[0])
cancer = cancer.sort(desc("count"))
display(cancer)

# COMMAND ----------


# Convert to Pandas DataFrame for plotting
cancer_df = cancer.toPandas()
# Create bar plot
plt.figure(figsize=(12, 6))
plt.bar(cancer_df["Question"], cancer_df["count"])
plt.xticks(rotation=90)
plt.title("Number of Records for Cancer Mortality")
plt.xlabel("Question")
plt.ylabel("Number of Records")
plt.show()

# COMMAND ----------

df.show()

# COMMAND ----------

drop = [7,8,15]
df = df.drop(*[df.columns[i] for i in drop])
df.show()

# COMMAND ----------

trainData, testData = df.randomSplit([0.7, 0.3], seed=42)


# COMMAND ----------

# MAGIC %md
# MAGIC # LinearRegression

# COMMAND ----------

from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, Imputer
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression
# Define the categorical and numerical column names
cat_cols = ["LocationAbbr", "LocationDesc", "DataSource", "Topic", "Question", "StratificationCategory1", "Stratification1"]
num_cols = ["YearStart", "YearEnd", "DataValue", "DataValueAlt", "LowConfidenceLimit", "HighConfidenceLimit"]

string_indexers = [StringIndexer(inputCol=col, outputCol=col+"_idx", handleInvalid="skip") for col in cat_cols]
one_hot_encoder = OneHotEncoder(inputCols=[col+"_idx" for col in cat_cols], outputCols=[col+"_ohe" for col in cat_cols])
imputer = Imputer(inputCols=num_cols, outputCols=[col+"_imputed" for col in num_cols])
assembler = VectorAssembler(inputCols=[col+"_ohe" for col in cat_cols] + [col+"_imputed" for col in num_cols], outputCol="features")
lr = LinearRegression(featuresCol="features", labelCol='DataValue')
stages = string_indexers + [one_hot_encoder, imputer, assembler]
# Create the pipeline
pipeline = Pipeline(stages=stages + [lr])
model = pipeline.fit(trainData)
predictions = model.transform(testData)

# COMMAND ----------

from pyspark.ml.feature import RFormula

rForm = RFormula(formula=" DataValue~ .", featuresCol="features", labelCol="DataValue", handleInvalid="skip")
pipeline = Pipeline(stages=[rForm, lr])
pipelineModel = pipeline.fit(trainData)
pred = pipelineModel.transform(testData)
pred.select("features", "DataValue", "prediction").show(5)

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator(labelCol='DataValue', rawPredictionCol='prediction')
accuracy = evaluator.evaluate(predictions)

print("Accuracy: {}".format(accuracy))

# COMMAND ----------

# MAGIC %md
# MAGIC # RandomForestRegressor

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, Imputer
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

location_indexer = StringIndexer(inputCol='LocationAbbr', outputCol='LocationAbbr_idx')
topic_indexer = StringIndexer(inputCol='Topic', outputCol='Topic_idx')
strat_category_indexer = StringIndexer(inputCol='StratificationCategory1', outputCol='StratificationCategory1_idx')
strat_indexer = StringIndexer(inputCol='Stratification1', outputCol='Stratification1_idx')
one_hot_encoder = OneHotEncoder(inputCols=['LocationAbbr_idx', 'Topic_idx', 'StratificationCategory1_idx', 'Stratification1_idx'], outputCols=['LocationAbbr_ohe', 'Topic_ohe', 'StratificationCategory1_ohe', 'Stratification1_ohe'])
imputer = Imputer(inputCols=['YearStart', 'YearEnd', 'DataValue', 'DataValueAlt', 'LowConfidenceLimit', 'HighConfidenceLimit'], outputCols=['YearStart_imputed', 'YearEnd_imputed', 'DataValue_imputed', 'DataValueAlt_imputed', 'LowConfidenceLimit_imputed', 'HighConfidenceLimit_imputed'])
assembler = VectorAssembler(inputCols=['LocationAbbr_ohe', 'Topic_ohe', 'StratificationCategory1_ohe', 'Stratification1_ohe', 'YearStart_imputed', 'YearEnd_imputed', 'DataValue_imputed', 'DataValueAlt_imputed', 'LowConfidenceLimit_imputed', 'HighConfidenceLimit_imputed'], outputCol='features')
rf = RandomForestRegressor(featuresCol='features', labelCol='DataValue')
# Define the stages of the pipeline
stages = [location_indexer, topic_indexer, strat_category_indexer, strat_indexer, one_hot_encoder, imputer, assembler]
pipeline = Pipeline(stages=stages + [rf])
model = pipeline.fit(trainData)
# Make predictions on the test data
predictions = model.transform(testData)

# COMMAND ----------

from pyspark.ml.feature import RFormula
rForm = RFormula(formula=" DataValue~ .", featuresCol="features", labelCol="DataValue", handleInvalid="skip")
pipeline = Pipeline(stages=[rForm, rf])
pipelineModel = pipeline.fit(trainData)
pred = pipelineModel.transform(testData)
pred.select("features", "DataValue", "prediction").show(5)

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator(labelCol='DataValue', rawPredictionCol='prediction')
accuracy = evaluator.evaluate(predictions)
print("Accuracy: {}".format(accuracy))

# COMMAND ----------

# MAGIC %md
# MAGIC The linear regression accuracy score you provided is quite high and close to one, which is unusual and may indicate overfitting. An accuracy score of 0.87 for the RandomForestRegressor shows that the model is working reasonably well.

# COMMAND ----------

# MAGIC %md
# MAGIC ## GraphFrames
# MAGIC

# COMMAND ----------

from graphframes import *
from pyspark.sql.functions import *

# Define vertices and edges DataFrame
vertices = df.selectExpr("LocationAbbr as id", "LocationDesc as location",)
edges = df.selectExpr("YearStart as src", "YearEnd as dst", "DataValue as weight")
g = GraphFrame(vertices, edges)

# COMMAND ----------

display(g.vertices)

# COMMAND ----------

display(g.edges)

# COMMAND ----------

print("Number of vertices: ", g.vertices.count())
print("Number of edges: ", g.edges.count())

# COMMAND ----------

result = g.vertices.groupBy("location").count()
display(result.orderBy(result["count"].desc()))

# COMMAND ----------

display(g.inDegrees)

# COMMAND ----------

from pyspark.sql.functions import desc
degrees = g.degrees.sort(desc("degree"))
degrees.show()

# COMMAND ----------

# MAGIC %md ## Subgraphs

# COMMAND ----------

sub_vertices = g.vertices.filter("location LIKE 'New%'")
# Define the subgraph edges
sub_edges = g.edges.filter("weight > 75 AND weight < 125")
subgraph = GraphFrame(sub_vertices, sub_edges)
subgraph.vertices.show()
subgraph.edges.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Connected components
# MAGIC

# COMMAND ----------

from pyspark.sql.functions import desc
from graphframes import GraphFrame
sc.setCheckpointDir("/tmp/graphframes-example-connected-components")
result = g.connectedComponents()
largest_component = result.groupBy("component").count().orderBy(desc("count")).first()["component"]
result=result.filter(result["component"] == largest_component).select("id").show()
result

# COMMAND ----------

# MAGIC %md
# MAGIC # labelPropagation

# COMMAND ----------

from pyspark.sql.functions import desc
from graphframes import GraphFrame
g = GraphFrame(vertices, edges)
result = g.labelPropagation(maxIter=2)
display(result.select("id", "label").orderBy("label"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## PageRank

# COMMAND ----------

from pyspark.sql.functions import desc
from graphframes import GraphFrame
#  PageRank with reset probability of 0.25 and tolerance of 0.01
pr = g.pageRank(resetProbability=0.25, tol=0.01)
pr.vertices.show()
