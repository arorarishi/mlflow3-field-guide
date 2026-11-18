# 04 · Datasets and lineage

**You will learn:** how to record which data a run used and how to spot "same name,
different data".

## Concepts

`mlflow.data` wraps your data in a **Dataset** object with a name, a **digest** (a content
hash), a **schema**, a **profile** (row count and similar), and a **source** (where it came
from). `mlflow.log_input(dataset, context=...)` attaches it to a run. The context is a label
such as `training`, `validation` or `evaluation`.

MLflow stores metadata about the data, **not the data**. Reproducibility therefore needs
an immutable or versioned source: a dated S3 prefix, a Delta table version, a DVC or lakeFS
commit, or a warehouse snapshot.

## Key APIs

```python
ds = mlflow.data.from_pandas(df, source="s3://bucket/2026-11-01/", name="train", targets="label")
mlflow.log_input(ds, context="training", tags={"version": "v3"})
mlflow.data.from_numpy(X, targets=y);   mlflow.data.from_spark(sdf)   # and more
run.inputs.dataset_inputs                 # read lineage back
```

In MLflow 3 you can also attach a dataset to a *metric*:
`mlflow.log_metric("f1", v, dataset=ds, model_id=...)` (chapter 09).

## Run it

`./book.sh run 04`. In the UI, the **Datasets used** column and the run page list the
datasets. Compare `ridge-v1` and `ridge-v2`: the train digest differs and the test digest does not.

## Exercises

1. Shuffle the rows of the training frame and check whether the digest changes.
2. Log a 100-row sample of the training data as an artifact for quick inspection.
