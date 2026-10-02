# Databricks notebook source
import numpy as np
import pandas as pd

FEATURE_TABLE = "workspace.default.telco_churn_features"
TARGET = "churn"
NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
EPS = 1e-6


def psi(expected: np.ndarray, actual: np.ndarray) -> float:
    expected, actual = expected + EPS, actual + EPS
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def numeric_psi(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    return psi(np.histogram(ref, edges)[0] / len(ref), np.histogram(cur, edges)[0] / len(cur))


def categorical_psi(ref: pd.Series, cur: pd.Series) -> float:
    expected = ref.astype(str).value_counts(normalize=True)
    actual = cur.astype(str).value_counts(normalize=True).reindex(expected.index, fill_value=0)
    return psi(expected.values, actual.values)


def drift_report(ref: pd.DataFrame, cur: pd.DataFrame) -> pd.Series:
    cols = [c for c in ref.columns if c not in ("customerID", TARGET)]
    return pd.Series(
        {c: numeric_psi(ref[c], cur[c]) if c in NUM_COLS else categorical_psi(ref[c], cur[c]) for c in cols}
    )


df = spark.table(FEATURE_TABLE).toPandas()
is_test = pd.util.hash_pandas_object(df["customerID"], index=False) % 5 == 0
reference, stable = df[~is_test], df[is_test]

drifted = stable.copy()
drifted["MonthlyCharges"] *= 1.25
flip = np.random.default_rng(42).random(len(drifted)) < 0.4
drifted.loc[flip, "Contract"] = "Month-to-month"

report = pd.DataFrame({"stable": drift_report(reference, stable), "drifted": drift_report(reference, drifted)})
report.sort_values("drifted", ascending=False).round(3).head(8)

# COMMAND ----------

from datetime import datetime, timezone

DRIFT_LOG = "workspace.default.telco_drift_log"
WATCH, ALERT = 0.1, 0.2
LOG_SCHEMA = (
    "checked_at string, batch string, max_psi double, top_feature string, "
    "watch_features string, alert_features string, retrain_needed boolean"
)


def evaluate_batch(name: str, scores: pd.Series) -> tuple:
    alerts = scores[scores > ALERT].index.tolist()
    watch = scores[(scores > WATCH) & (scores <= ALERT)].index.tolist()
    return (
        datetime.now(timezone.utc).isoformat(),
        name,
        float(scores.max()),
        scores.idxmax(),
        ",".join(watch),
        ",".join(alerts),
        bool(alerts),
    )


decisions = spark.createDataFrame([evaluate_batch(b, report[b]) for b in report.columns], LOG_SCHEMA)
decisions.write.mode("append").saveAsTable(DRIFT_LOG)
decisions.toPandas()

# COMMAND ----------

dbutils.widgets.dropdown("batch", "drifted", ["stable", "drifted"])
current = dbutils.widgets.get("batch")

retrain = bool(decisions.filter(f"batch = '{current}'").first()["retrain_needed"])
dbutils.jobs.taskValues.set(key="retrain_needed", value=retrain)

current, retrain
