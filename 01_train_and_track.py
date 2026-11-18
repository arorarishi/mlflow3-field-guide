"""Experiment tracking: a hyperparameter sweep as one parent run with nested child runs."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import CLASSIC_EXPERIMENT, SEED, TRACKING_URI, as_frame, load_splits

CANDIDATES = [
    ("logreg", {"C": 0.001}),
    ("logreg", {"C": 0.1}),
    ("logreg", {"C": 1.0}),
    ("random_forest", {"n_estimators": 10, "max_depth": 2}),
    ("random_forest", {"n_estimators": 100, "max_depth": 4}),
    ("random_forest", {"n_estimators": 300, "max_depth": None}),
]


def build(family, params):
    if family == "logreg":
        return make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED, **params)
        )
    return RandomForestClassifier(random_state=SEED, **params)


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(CLASSIC_EXPERIMENT)
    X_train, X_val, _, y_train, y_val, _ = load_splits()
    train_ds = mlflow.data.from_pandas(
        as_frame(X_train, y_train), name="breast_cancer_train", targets="target"
    )
    val_ds = mlflow.data.from_pandas(
        as_frame(X_val, y_val), name="breast_cancer_val", targets="target"
    )

    with mlflow.start_run(run_name="model-sweep"):
        mlflow.set_tag("purpose", "interview-demo")
        for family, params in CANDIDATES:
            name = f"{family}-" + "-".join(f"{k}={v}" for k, v in params.items())
            with mlflow.start_run(run_name=name, nested=True):
                mlflow.log_input(train_ds, context="training")
                mlflow.log_input(val_ds, context="validation")
                mlflow.log_param("family", family)
                mlflow.log_params(params)

                model = build(family, params).fit(X_train, y_train)
                preds = model.predict(X_val)
                metrics = {
                    "val_accuracy": accuracy_score(y_val, preds),
                    "val_precision": precision_score(y_val, preds),
                    "val_recall": recall_score(y_val, preds),
                    "val_f1": f1_score(y_val, preds),
                    "val_roc_auc": roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]),
                }

                # The signature is the input/output contract enforced at serving time.
                info = mlflow.sklearn.log_model(
                    model,
                    name="model",
                    signature=infer_signature(X_train, preds),
                    input_example=X_train.head(3),
                    params=params,
                    # skops is the safe default format. We trained this, so we trust this type.
                    skops_trusted_types=["sklearn.tree._tree.Tree"],
                )
                # Attaching metrics to the LoggedModel lets models be ranked directly.
                mlflow.log_metrics(metrics, model_id=info.model_id, dataset=val_ds)

                display = ConfusionMatrixDisplay.from_predictions(y_val, preds)
                mlflow.log_figure(display.figure_, "plots/confusion_matrix.png")
                plt.close(display.figure_)

                print(f"{name:45s} f1={metrics['val_f1']:.4f} auc={metrics['val_roc_auc']:.4f}")


if __name__ == "__main__":
    main()
