"""Chapter 01: where MLflow puts things.

Shows the tracking URI, the registry URI, and how one run splits into metadata (backend
store) and files (artifact store). Then repeats the run against a throwaway local SQLite
store to show that only the URI changes, not your code.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow
from mlflow import MlflowClient

from shared import TRACKING_URI, banner, setup, ui_link


def describe_run(label: str) -> None:
    with mlflow.start_run(run_name=f"hello-{label}") as run:
        mlflow.log_param("store", label)
        mlflow.log_metric("answer", 42)
        mlflow.log_text("artifacts are plain files", "notes/hello.txt")
    client = MlflowClient()
    print(f"tracking URI : {mlflow.get_tracking_uri()}")
    print(f"registry URI : {mlflow.get_registry_uri()}")
    print(f"run id       : {run.info.run_id}")
    print(f"artifact URI : {run.info.artifact_uri}")
    print(f"artifacts    : {[f.path for f in client.list_artifacts(run.info.run_id, 'notes')]}")


def main():
    banner("1. Remote tracking server (the normal setup)")
    experiment = setup("book-01-architecture")
    print(f"experiment   : {experiment.name} (id {experiment.experiment_id})")
    describe_run("server")
    print("mlflow-artifacts:/ means the server proxies artifact reads and writes for you.")
    print(f"Open {ui_link('experiments/' + experiment.experiment_id)}")

    banner("2. Same code against a local SQLite store (no server)")
    with tempfile.TemporaryDirectory() as tmp:
        mlflow.set_tracking_uri(f"sqlite:///{tmp}/local.db")
        # With no server, artifacts default to ./mlruns. Pin them inside the temp folder.
        experiment_id = mlflow.create_experiment("book-01-local", artifact_location=f"{tmp}/artifacts")
        mlflow.set_experiment(experiment_id=experiment_id)
        describe_run("local-sqlite")
        print("Without a server, the artifact URI is a local path the client writes directly.")

    mlflow.set_tracking_uri(TRACKING_URI)


if __name__ == "__main__":
    main()
