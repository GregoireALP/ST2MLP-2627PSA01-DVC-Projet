"""Download the latest trained model from DagsHub (MLflow) into $MODEL_DIR/model.

Run at Docker build time. Requires:
  MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD (DagsHub username + access token)
"""
import os
import shutil
import sys

import mlflow
from mlflow.tracking import MlflowClient

TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "https://dagshub.com/GregoireALP/ST2MLP-2627PSA01-DVC-Projet.mlflow",
)
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "RandomForest_V1.0.1")
DEST = os.getenv("MODEL_DIR", "model")


def fail(msg: str):
    print(f"\n[download_model] ERROR: {msg}\n", file=sys.stderr)
    sys.exit(1)


# 1. Credentials
user = os.getenv("MLFLOW_TRACKING_USERNAME", "").strip()
token = os.getenv("MLFLOW_TRACKING_PASSWORD", "").strip()
if not user or not token:
    fail(
        "DagsHub credentials are empty "
        f"(username set: {bool(user)}, token set: {bool(token)}). "
        "Add the DAGSHUB_USERNAME and DAGSHUB_TOKEN secrets in GitHub "
        "(Settings > Secrets and variables > Actions)."
    )
os.environ["MLFLOW_TRACKING_USERNAME"] = user
os.environ["MLFLOW_TRACKING_PASSWORD"] = token

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()
print(f"[download_model] Tracking URI: {TRACKING_URI} (user: {user})")

# 2. Experiment (also validates authentication)
try:
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
except Exception as e:
    fail(f"Cannot reach MLflow on DagsHub (bad username/token?): {e}")
if experiment is None:
    names = [e.name for e in client.search_experiments()]
    fail(f"Experiment '{EXPERIMENT_NAME}' not found. Available: {names}")

# 3. Most recent finished runs, newest first
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="attributes.status = 'FINISHED'",
    order_by=["attributes.start_time DESC"],
    max_results=10,
)
if not runs:
    fail(f"No finished run in experiment '{EXPERIMENT_NAME}'")


def candidate_uris(run_id: str):
    # MLflow 3 stores models as "logged models" linked to the run
    try:
        for m in client.search_logged_models(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"source_run_id = '{run_id}'",
        ):
            yield f"models:/{m.model_id}"
    except Exception as e:
        print(f"[download_model] logged-model lookup unavailable: {e}")
    # Classic location: run artifacts
    yield f"runs:/{run_id}/model"


shutil.rmtree(DEST, ignore_errors=True)
target = os.path.join(DEST, "model")

for run in runs:
    print(f"[download_model] Trying run {run.info.run_name} ({run.info.run_id}) "
          f"metrics={run.data.metrics}")
    for uri in candidate_uris(run.info.run_id):
        try:
            os.makedirs(target, exist_ok=True)
            path = mlflow.artifacts.download_artifacts(artifact_uri=uri, dst_path=target)
            # models:/ downloads the MLmodel dir itself; runs:/.../model adds a "model" subdir
            if not os.path.exists(os.path.join(target, "MLmodel")):
                inner = os.path.join(path)
                if os.path.exists(os.path.join(inner, "MLmodel")) and inner != target:
                    tmp = target + "_tmp"
                    shutil.move(inner, tmp)
                    shutil.rmtree(target)
                    shutil.move(tmp, target)
            if os.path.exists(os.path.join(target, "MLmodel")):
                print(f"[download_model] OK: {uri} -> {target}")
                sys.exit(0)
            print(f"[download_model] {uri}: no MLmodel file found")
        except Exception as e:
            print(f"[download_model] {uri} failed: {e}")
        shutil.rmtree(target, ignore_errors=True)

fail("No downloadable model found in the 10 most recent runs.")
