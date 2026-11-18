# 06 · System metrics and logging performance

**You will learn:** how to record hardware usage alongside a run, and how to log
large volumes of metrics without slowing down training.

## System metrics

`start_run(log_system_metrics=True)`, or `mlflow.enable_system_metrics_logging()` globally,
starts a background sampler. It records CPU utilisation, memory, disk, network, and GPU
utilisation and memory (needs `nvidia-ml-py`). Metrics are logged under `system/`, and the
UI shows them on the run's **System metrics** tab.

```python
mlflow.set_system_metrics_sampling_interval(1)          # seconds
mlflow.set_system_metrics_samples_before_logging(1)
mlflow.set_system_metrics_node_id("worker-3")           # multi-node jobs
```

## Logging fast

| Method | When |
|---|---|
| `log_metric` per point | simple scripts, few points |
| `MlflowClient().log_batch(run_id, metrics=[...])` | you already have many points (up to 1000 metrics per call) |
| `log_metric(..., synchronous=False)` or `mlflow.config.enable_async_logging()` | inside a training loop. Logging happens in a background thread. Call `mlflow.flush_async_logging()` before you read the results back. |

The example measures all three. On a local server, a batch is roughly 100 times faster than
one call per point.

## Run it

`./book.sh run 06`. Open run `busy-cpu`, then the **System metrics** tab.

## Exercises

1. Turn on async logging globally and re-time the one-by-one loop.
2. Log `system/` metrics during an Optuna study (chapter 13). Which trials were CPU-bound?
