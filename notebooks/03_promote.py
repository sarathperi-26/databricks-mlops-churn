# Databricks notebook source
import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.metrics import roc_auc_score

FEATURE_TABLE = "workspace.default.telco_churn_features"
MODEL_NAME = "workspace.default.telco_churn_model"
TARGET = "churn"

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

df = spark.table(FEATURE_TABLE).toPandas()
test = df[pd.util.hash_pandas_object(df["customerID"], index=False) % 5 == 0]
X, y = test.drop(columns=["customerID", TARGET]), test[TARGET]


def score(alias: str) -> tuple[str, float] | None:
    try:
        mv = client.get_model_version_by_alias(MODEL_NAME, alias)
    except MlflowException:
        return None
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{alias}")
    return mv.version, roc_auc_score(y, model.predict_proba(X)[:, 1])


champion, challenger = score("champion"), score("challenger")
champion, challenger

# COMMAND ----------

from datetime import datetime, timezone

MIN_DELTA = 0.005
PROMOTION_LOG = "workspace.default.telco_promotion_log"
LOG_SCHEMA = (
    "decided_at string, champion_version string, challenger_version string, "
    "champion_auc double, challenger_auc double, min_delta double, promoted boolean"
)

if challenger is None:
    raise ValueError("No challenger registered. Run 02_train first.")

ch_version, ch_auc = challenger
champ_version, champ_auc = champion if champion else (None, None)
promote = champ_auc is None or ch_auc >= champ_auc + MIN_DELTA

if promote:
    client.set_registered_model_alias(MODEL_NAME, "champion", ch_version)
client.delete_registered_model_alias(MODEL_NAME, "challenger")

decision = {
    "decided_at": datetime.now(timezone.utc).isoformat(),
    "champion_version": champ_version,
    "challenger_version": ch_version,
    "champion_auc": champ_auc,
    "challenger_auc": ch_auc,
    "min_delta": MIN_DELTA,
    "promoted": promote,
}
spark.createDataFrame([tuple(decision.values())], LOG_SCHEMA).write.mode("append").saveAsTable(PROMOTION_LOG)

decision