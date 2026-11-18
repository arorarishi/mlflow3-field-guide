# 02 · Tracking basics: runs, params, metrics, tags, artifacts

**You will learn:** the core logging API and the data model behind every run.

## Data model

| Thing | Type | Mutable? | Use for |
|---|---|---|---|
| Experiment | named container | renameable | one problem or project |
| Run | one execution | status changes | one training job, one sweep trial |
| Param | string key/value | **write-once** | inputs: hyperparameters, config |
| Metric | float + step + timestamp | append-only history | outputs over time: loss, accuracy |
| Tag | string key/value | yes | labels for search: team, stage, git info |
| Artifact | file | overwrite by path | models, plots, reports, data samples |

MLflow sets system tags automatically, such as `mlflow.runName`, `mlflow.user`,
`mlflow.source.name`, and `mlflow.source.git.commit` when the code runs inside a git repository.

## Key APIs

```python
with mlflow.start_run(run_name="basics") as run:
    mlflow.log_param("lr", 0.01);           mlflow.log_params({...})
    mlflow.log_metric("loss", v, step=i);   mlflow.log_metrics({...})
    mlflow.set_tag("team", "risk");         mlflow.set_tags({...})
    mlflow.log_text(s, "notes.txt");        mlflow.log_dict(d, "config.json")
    mlflow.log_figure(fig, "plot.png");     mlflow.log_table(df, "table.json")
    mlflow.log_artifact("file.md", "reports");  mlflow.log_artifacts("dir/")

with mlflow.start_run(run_id=run.info.run_id):   # resume and append
    ...
```

Read back with `MlflowClient().get_run(id)`, `get_metric_history(id, key)`, and
`mlflow.artifacts.load_dict/load_text/download_artifacts`.

## Run it

`./book.sh run 02`. In the UI, open run `basics` and check the **Model metrics** chart for
`train_loss`, the **Artifacts** tab, and the failed run `crashes`.

## Pitfalls

* Logging a param twice with a *different* value raises an error. Params describe the run's
  inputs, and those do not change. Put changing values in metrics.
* An exception inside `with start_run()` marks the run `FAILED`. Without the context manager,
  you must call `mlflow.end_run()` yourself.
* Metric keys are free text. Agree on a naming scheme (`val_f1`, not sometimes `f1_val`).

## Exercises

1. Log a metric with `step` and `timestamp`, and switch the chart's x-axis between them.
2. Write a helper that logs `git diff` as an artifact when the working tree is dirty.
