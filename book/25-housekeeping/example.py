"""Chapter 25: housekeeping.

Deleting in MLflow is soft: runs and experiments move to a 'deleted' state and can be
restored. `mlflow gc` removes them for good, including their artifacts. This chapter
uses a throwaway local store (no server), so nothing here touches your real data.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow
from mlflow import MlflowClient
from mlflow.entities import ViewType

from shared import TRACKING_URI, banner, cli_env, run_cli


def size(path: Path) -> str:
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return f"{total / 1024:.0f} KiB in {sum(1 for d in path.glob('*/*') if d.is_dir())} run folders"


def main():
    with tempfile.TemporaryDirectory() as tmp:
        store = f"sqlite:///{tmp}/mlflow.db"
        artifacts = Path(tmp) / "artifacts"
        mlflow.set_tracking_uri(store)
        env = cli_env(MLFLOW_TRACKING_URI=store)
        client = MlflowClient()

        banner("Create an experiment with five runs, each with a 200 KiB artifact")
        experiment_id = mlflow.create_experiment("cleanup-demo", artifact_location=str(artifacts))
        run_ids = []
        for i in range(5):
            with mlflow.start_run(experiment_id=experiment_id, run_name=f"run-{i}") as run:
                mlflow.log_metric("score", i)
                mlflow.log_text("x" * 200_000, "blob.txt")
                run_ids.append(run.info.run_id)
        print("artifacts:", size(artifacts))

        banner("Soft delete three runs")
        for run_id in run_ids[:3]:
            client.delete_run(run_id)
        deleted = mlflow.search_runs([experiment_id], run_view_type=ViewType.DELETED_ONLY, output_format="list")
        print(f"deleted runs: {len(deleted)}. Artifacts are still there: {size(artifacts)}")

        banner("Export what is left before cleaning up")
        run_cli("experiments", "csv", "-x", experiment_id, "-o", f"{tmp}/runs.csv", env=env)
        print(Path(f"{tmp}/runs.csv").read_text().splitlines()[0][:100], "...")

        banner("Garbage-collect deleted runs (permanent)")
        run_cli("gc", "--backend-store-uri", store, "--older-than", "0s", env=env,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("after gc:", size(artifacts))
        left = mlflow.search_runs([experiment_id], run_view_type=ViewType.ALL, output_format="list")
        print(f"runs left in any state: {len(left)}")

        banner("Experiments: delete, restore, rename")
        client.delete_experiment(experiment_id)
        print("state:", client.get_experiment(experiment_id).lifecycle_stage)
        client.restore_experiment(experiment_id)
        client.rename_experiment(experiment_id, "cleanup-demo-archived")
        print("state:", client.get_experiment(experiment_id).lifecycle_stage, "| name:", client.get_experiment(experiment_id).name)

        banner("Schema migrations after upgrading MLflow")
        run_cli("db", "upgrade", store, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("`mlflow db upgrade <uri>` is idempotent. Back up the database first.")

        banner("Environment report for bug reports")
        doctor = run_cli("doctor", env=env, capture_output=True)
        print("\n".join(doctor.stdout.splitlines()[:8]))

    mlflow.set_tracking_uri(TRACKING_URI)


if __name__ == "__main__":
    main()
