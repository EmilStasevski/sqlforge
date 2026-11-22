"""Register the QLoRA adapter in the MLflow model registry (model "sqlforge-qlora").

Attaches the adapter directory as artifacts to the training run (found by run name) and creates a registered
model version from it, tagged with the eval results from results/eval_qlora_*.json.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

NAME = "sqlforge-qlora"


def main() -> None:
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    client = MlflowClient()
    exp = client.get_experiment_by_name("sqlforge-train")
    runs = client.search_runs(
        [exp.experiment_id],
        "attributes.run_name = 'qlora'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise SystemExit("no MLflow run named 'qlora' in experiment sqlforge-train")
    run = runs[0]
    adapter = Path("runs/qlora/adapter")
    with mlflow.start_run(run_id=run.info.run_id):
        mlflow.log_artifacts(str(adapter), artifact_path="adapter")
        for p in Path("results").glob("eval_qlora_*.json"):
            d = json.loads(p.read_text())
            mlflow.log_metric(f"{d['split']}_execution_accuracy", d["execution_accuracy"])
    try:
        client.create_registered_model(
            NAME, description="Qwen2.5-Coder-7B QLoRA adapter for DE/EN text-to-DuckDB-SQL"
        )
    except mlflow.exceptions.MlflowException:
        pass  # already exists
    mv = client.create_model_version(NAME, f"runs:/{run.info.run_id}/adapter", run_id=run.info.run_id)
    meta = json.loads((adapter / "run_meta.json").read_text())
    for k in ("git_sha", "base_model"):
        client.set_model_version_tag(NAME, mv.version, k, str(meta[k]))
    for p in Path("results").glob("eval_qlora_*.json"):
        d = json.loads(p.read_text())
        client.set_model_version_tag(NAME, mv.version, f"EX_{d['split']}", str(d["execution_accuracy"]))
    print(f"registered {NAME} v{mv.version} from run {run.info.run_id}")


if __name__ == "__main__":
    main()
