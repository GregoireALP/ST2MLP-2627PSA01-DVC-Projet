"""Download the latest trained model from DagsHub (MLflow) into ./model.

Run at Docker build time. Requires:
  MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD (DagsHub user + token)
"""
import os
import shutil

import mlflow

TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "https://dagshub.com/GregoireALP/ST2MLP-2627PSA01-DVC-Projet.mlflow",
)
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "RandomForest_V1.0.1")
DEST = os.getenv("MODEL_DIR", "model")

mlflow.set_tracking_uri(TRACKING_URI)

experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    raise SystemExit(f"Experiment '{EXPERIMENT_NAME}' not found on {TRACKING_URI}")

# Most recent successful run of the experiment
runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="attributes.status = 'FINISHED'",
    order_by=["attributes.start_time DESC"],
    max_results=1,
    output_format="list",
)
if not runs:
    raise SystemExit(f"No finished run in experiment '{EXPERIMENT_NAME}'")

run = runs[0]
print(f"Latest run: {run.info.run_name} ({run.info.run_id}) metrics={run.data.metrics}")

shutil.rmtree(DEST, ignore_errors=True)
path = mlflow.artifacts.download_artifacts(
    artifact_uri=f"runs:/{run.info.run_id}/model", dst_path=DEST
)
print(f"Model downloaded to {path}")
