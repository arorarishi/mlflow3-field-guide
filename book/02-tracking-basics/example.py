"""Chapter 02: the core tracking API.

Runs, params, metrics with steps, tags, every kind of artifact, run status, and
resuming a finished run to add more data.
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient

from shared import banner, setup


def main():
    setup("book-02-tracking-basics")

    banner("Start a run and log the basics")
    with mlflow.start_run(run_name="basics", description="Tour of the tracking API") as run:
        # Params are inputs. They are write-once: logging the same key with a new value fails.
        mlflow.log_param("learning_rate", 0.01)
        mlflow.log_params({"optimizer": "adam", "epochs": 20, "layers": [64, 32]})

        # Metrics are numbers that can change over time. `step` gives you a curve.
        for epoch in range(20):
            loss = 1.0 / (epoch + 1) + np.random.default_rng(epoch).normal(0, 0.01)
            mlflow.log_metric("train_loss", loss, step=epoch)
        mlflow.log_metrics({"final_accuracy": 0.93, "final_f1": 0.91})

        # Tags are free-form, mutable labels. Use them to filter and to group runs.
        mlflow.set_tags({"team": "risk", "stage": "exploration", "owner": "book-reader"})

        # Artifacts: any file. The helpers write the file for you.
        mlflow.log_text("Trained on synthetic data.", "notes/summary.txt")
        mlflow.log_dict({"threshold": 0.5, "classes": ["no", "yes"]}, "config/config.json")
        mlflow.log_table(
            pd.DataFrame({"feature": ["age", "income"], "importance": [0.7, 0.3]}),
            "tables/importance.json",
        )
        figure, axis = plt.subplots()
        axis.plot(range(20), [1 / (e + 1) for e in range(20)])
        axis.set_title("loss")
        mlflow.log_figure(figure, "plots/loss.png")
        plt.close(figure)

        # Or log files and folders you wrote yourself.
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "report.md"
            report.write_text("# Report\nEverything is fine.\n")
            mlflow.log_artifact(str(report), artifact_path="reports")

        run_id = run.info.run_id
        print(f"run_id = {run_id}")

    banner("Read it back")
    client = MlflowClient()
    finished = client.get_run(run_id)
    print(f"status  : {finished.info.status}")
    print(f"params  : {finished.data.params}")
    print(f"metrics : {finished.data.metrics}  (latest value of each)")
    history = client.get_metric_history(run_id, "train_loss")
    print(f"train_loss has {len(history)} points, first={history[0].value:.3f} last={history[-1].value:.3f}")
    config = mlflow.artifacts.load_dict(f"runs:/{run_id}/config/config.json")
    print(f"config.json -> {json.dumps(config)}")

    banner("Resume the finished run and add to it")
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("post_hoc_check", 1.0)
        mlflow.set_tag("stage", "reviewed")
    print(f"tags now: { {k: v for k, v in client.get_run(run_id).data.tags.items() if not k.startswith('mlflow.')} }")

    banner("A failing run is recorded as FAILED")
    try:
        with mlflow.start_run(run_name="crashes") as bad:
            mlflow.log_param("will_fail", True)
            raise ValueError("simulated training crash")
    except ValueError:
        pass
    print(f"status of 'crashes': {client.get_run(bad.info.run_id).info.status}")


if __name__ == "__main__":
    main()
