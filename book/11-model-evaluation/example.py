"""Chapter 11: evaluating classic ML models.

mlflow.models.evaluate computes a standard metric set and plots for a model type, logs
them to the run, and accepts your own metrics. validate_evaluation_results then turns
"is the candidate good enough, and better than the baseline?" into a pass/fail check.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow
import numpy as np
from mlflow.models import MetricThreshold, make_metric
from mlflow.models.evaluation.validation import ModelValidationFailedException
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from shared import banner, classification_splits, setup, with_target


def missed_cancer_rate(predictions, targets, metrics):
    """Business metric: the share of malignant (0) cases predicted benign (1)."""
    malignant = np.asarray(targets) == 0
    return float(np.mean(np.asarray(predictions)[malignant] == 1))


MISSED = make_metric(eval_fn=missed_cancer_rate, greater_is_better=False, name="missed_cancer_rate")


def evaluate(model, name, eval_df):
    with mlflow.start_run(run_name=name):
        info = mlflow.sklearn.log_model(model, name="model", input_example=eval_df.drop(columns="target").head(2))
        return mlflow.models.evaluate(
            info.model_uri,
            eval_df,
            targets="target",
            model_type="classifier",
            extra_metrics=[MISSED],
            evaluators=["default"],
        )


def main():
    setup("book-11-evaluation")
    X_train, X_val, _, y_train, y_val, _ = classification_splits()
    eval_df = with_target(X_val, y_val)

    banner("Evaluate a baseline and a candidate")
    baseline = evaluate(DummyClassifier(strategy="most_frequent").fit(X_train, y_train), "baseline", eval_df)
    candidate = evaluate(
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X_train, y_train), "candidate", eval_df
    )
    keys = ["accuracy_score", "f1_score", "roc_auc", "missed_cancer_rate"]
    for label, result in (("baseline", baseline), ("candidate", candidate)):
        print(f"{label:9s}", {k: round(result.metrics[k], 3) for k in keys if k in result.metrics})
    print(f"artifacts logged: {sorted(candidate.artifacts)[:6]}")

    banner("Gate: absolute thresholds and improvement over baseline")
    thresholds = {
        "f1_score": MetricThreshold(threshold=0.9, min_absolute_change=0.05, greater_is_better=True),
        "missed_cancer_rate": MetricThreshold(threshold=0.1, greater_is_better=False),
    }
    try:
        mlflow.validate_evaluation_results(thresholds, candidate, baseline)
        print("candidate PASSED validation")
    except ModelValidationFailedException as error:
        print(f"candidate FAILED: {error}")

    banner("The baseline cannot pass the same gate")
    try:
        mlflow.validate_evaluation_results(thresholds, baseline, None)
    except ModelValidationFailedException as error:
        print(f"baseline FAILED as expected:\n{str(error)[:300]}")


if __name__ == "__main__":
    main()
