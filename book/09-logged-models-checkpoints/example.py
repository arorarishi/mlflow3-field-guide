"""Chapter 09: LoggedModel, the MLflow 3 model entity.

In MLflow 3 a model is a first-class object with its own ID (m-...), params, metrics and
tags, independent of the run that produced it. Here a neural net trains for 30 epochs and
saves a checkpoint every 5. Each checkpoint gets its own validation metrics, so you can
pick the best checkpoint with one query instead of digging through run artifacts.
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow
import numpy as np
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from shared import banner, classification_splits, setup

warnings.filterwarnings("ignore")
TRUSTED = "sklearn.neural_network._stochastic_optimizers.AdamOptimizer"


def main():
    experiment = setup("book-09-logged-models")
    X_train, X_val, _, y_train, y_val, _ = classification_splits()
    scaler = StandardScaler().fit(X_train)
    Xt, Xv = scaler.transform(X_train), scaler.transform(X_val)
    val_ds = mlflow.data.from_numpy(Xv, targets=np.asarray(y_val), name="val")

    banner("Train with per-epoch metrics and periodic checkpoints")
    net = MLPClassifier(hidden_layer_sizes=(32,), learning_rate_init=0.01, random_state=0)
    with mlflow.start_run(run_name="mlp-30-epochs") as run:
        mlflow.log_params({"hidden": 32, "lr": 0.01})
        for epoch in range(1, 31):
            net.partial_fit(Xt, y_train, classes=[0, 1])
            val_loss = log_loss(y_val, net.predict_proba(Xv))
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            if epoch % 5 == 0:
                info = mlflow.sklearn.log_model(
                    net,
                    name=f"checkpoint-{epoch:02d}",
                    step=epoch,
                    # skops refuses unknown types on load. We built this net, so we trust it.
                    skops_trusted_types=[TRUSTED],
                )
                mlflow.log_metrics(
                    {"val_loss": val_loss, "val_accuracy": accuracy_score(y_val, net.predict(Xv))},
                    step=epoch,
                    model_id=info.model_id,
                    dataset=val_ds,
                )
                print(f"epoch {epoch:2d}: val_loss={val_loss:.4f} -> {info.model_id}")

    banner("Rank checkpoints by their own metrics")
    ranked = mlflow.search_logged_models(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"source_run_id = '{run.info.run_id}'",
        order_by=[{"field_name": "metrics.val_loss", "ascending": True}],
        output_format="list",
    )
    for model in ranked:
        metrics = {m.key: round(m.value, 4) for m in model.metrics}
        print(f"{model.name}  {model.model_id}  {metrics}")
    best = ranked[0]
    print(f"best checkpoint: {best.name}, load with models:/{best.model_id}")
    mlflow.set_logged_model_tags(best.model_id, {"selected": "true"})

    banner("A LoggedModel for something MLflow did not package")
    external = mlflow.create_external_model(
        name="vendor-scoring-api", model_type="external", tags={"vendor": "acme"}
    )
    mlflow.log_metrics({"vendor_auc": 0.91}, model_id=external.model_id)
    print(f"external model {external.model_id}: metrics and lineage, no files. Useful for APIs you call.")


if __name__ == "__main__":
    main()
