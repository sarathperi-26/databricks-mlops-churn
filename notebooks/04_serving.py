# Databricks notebook source
import time

import pandas as pd
from mlflow.deployments import get_deploy_client

FEATURE_TABLE = "workspace.default.telco_churn_features"
ENDPOINT = "telco-churn-endpoint"
TARGET = "churn"

deploy = get_deploy_client("databricks")

df = spark.table(FEATURE_TABLE).toPandas()
test = df[pd.util.hash_pandas_object(df["customerID"], index=False) % 5 == 0]
sample = test.drop(columns=["customerID", TARGET]).head(5)

start = time.perf_counter()
response = deploy.predict(
    endpoint=ENDPOINT,
    inputs={"dataframe_split": sample.to_dict(orient="split")},
)
elapsed = time.perf_counter() - start

pd.DataFrame({"predicted": response["predictions"], "actual": test[TARGET].head(5).values}), round(elapsed, 2)

# COMMAND ----------

rows = test.drop(columns=["customerID", TARGET]).head(20)
latencies = []

for i in range(len(rows)):
    start = time.perf_counter()
    deploy.predict(
        endpoint=ENDPOINT,
        inputs={"dataframe_split": rows.iloc[[i]].to_dict(orient="split")},
    )
    latencies.append(time.perf_counter() - start)

s = pd.Series(latencies)
round(s.median(), 3), round(s.quantile(0.95), 3), round(s.max(), 3)