import sys

import mlflow

MODEL_NAME = "workspace.default.telco_churn_model"


def main(alias: str = "champion", dst: str = "model") -> None:
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    path = mlflow.artifacts.download_artifacts(f"models:/{MODEL_NAME}@{alias}", dst_path=dst)
    version = mlflow.MlflowClient().get_model_version_by_alias(MODEL_NAME, alias).version
    print(f"{MODEL_NAME}@{alias} -> version {version} at {path}")


if __name__ == "__main__":
    main(*sys.argv[1:])
