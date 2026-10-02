# Databricks notebook source
import pandas as pd

CATALOG, SCHEMA = "workspace", "default"
RAW_TABLE = f"{CATALOG}.{SCHEMA}.telco_churn_raw"
FEATURE_TABLE = f"{CATALOG}.{SCHEMA}.telco_churn_features"
TARGET = "churn"


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=str.strip)
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"].astype(str).str.strip(), errors="coerce"
    ).fillna(0.0)
    df[TARGET] = df.pop("Churn").str.strip().eq("Yes").astype(int)
    return df.drop_duplicates("customerID")


features = clean(spark.table(RAW_TABLE).toPandas())

(
    spark.createDataFrame(features)
    .write.mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(FEATURE_TABLE)
)

features.shape, round(features[TARGET].mean(), 3), int(features.isna().sum().sum())
