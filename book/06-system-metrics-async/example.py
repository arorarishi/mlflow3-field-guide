"""Chapter 06: system metrics and logging performance.

System metrics record CPU, memory, disk and network (and GPU when present) during a run.
Then we compare three ways to log many metrics: one call each, one batch, and async.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow
import numpy as np
from mlflow import MlflowClient
from mlflow.entities import Metric

from shared import banner, setup

N = 200


def main():
    setup("book-06-system-metrics")
    client = MlflowClient()

    banner("System metrics while a workload runs")
    mlflow.set_system_metrics_sampling_interval(1)
    mlflow.set_system_metrics_samples_before_logging(1)
    with mlflow.start_run(run_name="busy-cpu", log_system_metrics=True) as run:
        deadline = time.time() + 6
        while time.time() < deadline:
            np.linalg.svd(np.random.rand(300, 300))
    time.sleep(1)
    system = {k: round(v, 1) for k, v in mlflow.get_run(run.info.run_id).data.metrics.items() if k.startswith("system/")}
    print(f"{len(system)} system metrics, e.g. {dict(list(system.items())[:4])}")

    banner(f"Logging {N} metric points three ways")
    with mlflow.start_run(run_name="one-by-one") as run:
        start = time.perf_counter()
        for step in range(N):
            mlflow.log_metric("loss", 1 / (step + 1), step=step)
        print(f"one call per point : {time.perf_counter() - start:6.2f}s")

    with mlflow.start_run(run_name="batched") as run:
        start = time.perf_counter()
        now = int(time.time() * 1000)
        metrics = [Metric("loss", 1 / (step + 1), now, step) for step in range(N)]
        client.log_batch(run.info.run_id, metrics=metrics)  # up to 1000 metrics per call
        print(f"log_batch          : {time.perf_counter() - start:6.2f}s")

    with mlflow.start_run(run_name="async") as run:
        start = time.perf_counter()
        for step in range(N):
            mlflow.log_metric("loss", 1 / (step + 1), step=step, synchronous=False)
        queued = time.perf_counter() - start
        mlflow.flush_async_logging()
        print(f"async (queued)     : {queued:6.2f}s, flushed at {time.perf_counter() - start:.2f}s")
    points = len(client.get_metric_history(run.info.run_id, "loss"))
    print(f"async run stored {points} points. Turn async on globally with mlflow.config.enable_async_logging().")


if __name__ == "__main__":
    main()
