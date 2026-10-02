# Databricks notebook source
import mlflow
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURE_TABLE = "workspace.default.telco_churn_features"
TARGET = "churn"
SEED = 42
NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

user = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{user}/telco-churn")
mlflow.sklearn.autolog(disable=True)

df = spark.table(FEATURE_TABLE).toPandas()
is_test = pd.util.hash_pandas_object(df["customerID"], index=False) % 5 == 0
train, test = df[~is_test], df[is_test]

feature_cols = [c for c in df.columns if c not in ("customerID", TARGET)]
cat_cols = [c for c in feature_cols if c not in NUM_COLS]

len(train), len(test), round(train[TARGET].mean(), 3), round(test[TARGET].mean(), 3)

# COMMAND ----------

def build(estimator) -> Pipeline:
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), NUM_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )
    return Pipeline([("pre", pre), ("clf", estimator)])


def evaluate(model, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "roc_auc": roc_auc_score(y, proba),
        "f1": f1_score(y, pred),
        "precision": precision_score(y, pred),
        "recall": recall_score(y, pred),
    }


candidates = {
    "logreg": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "random_forest": RandomForestClassifier(
        n_estimators=300, min_samples_leaf=5, class_weight="balanced",
        random_state=SEED, n_jobs=-1,
    ),
    "grad_boost": GradientBoostingClassifier(random_state=SEED),
}

list(candidates)

# COMMAND ----------

results = {}
for name, estimator in candidates.items():
    with mlflow.start_run(run_name=name) as run:
        model = build(estimator).fit(train[feature_cols], train[TARGET])
        metrics = evaluate(model, test[feature_cols], test[TARGET])

        mlflow.log_params({"model": name, "train_rows": len(train), "test_rows": len(test)})
        mlflow.log_params(estimator.get_params())
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=infer_signature(test[feature_cols], model.predict(test[feature_cols])),
            input_example=test[feature_cols].head(3),
        )
        results[name] = {"run_id": run.info.run_id, **metrics}

pd.DataFrame(results).T.sort_values("roc_auc", ascending=False)

# COMMAND ----------

from mlflow import MlflowClient

MODEL_NAME = "workspace.default.telco_churn_model"
mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

best_name, best = max(results.items(), key=lambda kv: kv[1]["roc_auc"])
version = mlflow.register_model(f"runs:/{best['run_id']}/model", MODEL_NAME)
client.set_registered_model_alias(MODEL_NAME, "challenger", version.version)
client.set_model_version_tag(MODEL_NAME, version.version, "roc_auc", f"{best['roc_auc']:.4f}")

best_name, version.version