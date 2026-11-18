"""Chapter 04: datasets and lineage.

MLflow records *which* data a run used (name, digest, schema, source), not the data
itself. A changed digest tells you two runs did not train on the same data.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

from shared import banner, regression_splits, setup, with_target

SOURCE = "https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html"


def main():
    setup("book-04-datasets")
    X_train, X_test, y_train, y_test = regression_splits()

    banner("Describe the data")
    train = mlflow.data.from_pandas(
        with_target(X_train, y_train), source=SOURCE, name="diabetes-train", targets="target"
    )
    test = mlflow.data.from_pandas(
        with_target(X_test, y_test), source=SOURCE, name="diabetes-test", targets="target"
    )
    print(f"name={train.name} digest={train.digest} rows={train.profile['num_rows']}")
    print(f"schema columns: {[c.name for c in train.schema.inputs][:5]} ...")

    banner("Attach datasets to runs, with a context")
    with mlflow.start_run(run_name="ridge-v1") as run_v1:
        mlflow.log_input(train, context="training", tags={"version": "v1"})
        mlflow.log_input(test, context="evaluation")
        model = Ridge(alpha=1.0).fit(X_train, y_train)
        mlflow.log_metric("test_rmse", mean_squared_error(y_test, model.predict(X_test)) ** 0.5)

    # Same name, different contents: drop the 50 highest-BMI rows (a "data fix").
    keep = X_train["bmi"].rank(ascending=False) > 50
    train_v2 = mlflow.data.from_pandas(
        with_target(X_train[keep], y_train[keep]), source=SOURCE, name="diabetes-train", targets="target"
    )
    with mlflow.start_run(run_name="ridge-v2") as run_v2:
        mlflow.log_input(train_v2, context="training", tags={"version": "v2"})
        mlflow.log_input(test, context="evaluation")
        model = Ridge(alpha=1.0).fit(X_train[keep], y_train[keep])
        mlflow.log_metric("test_rmse", mean_squared_error(y_test, model.predict(X_test)) ** 0.5)

    banner("Read lineage back")
    for run in (run_v1, run_v2):
        inputs = mlflow.get_run(run.info.run_id).inputs.dataset_inputs
        for item in inputs:
            context = {t.key: t.value for t in item.tags}.get("mlflow.data.context")
            print(f"{run.info.run_name}: {item.dataset.name:15s} digest={item.dataset.digest} context={context}")
    print("Same test digest in both runs: a fair comparison. Different train digests: different data.")

    banner("Numpy arrays work too")
    array_ds = mlflow.data.from_numpy(np.asarray(X_train), targets=np.asarray(y_train), name="diabetes-np")
    print(f"numpy dataset digest={array_ds.digest}")


if __name__ == "__main__":
    main()
