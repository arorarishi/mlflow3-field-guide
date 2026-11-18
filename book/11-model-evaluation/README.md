# 11 · Evaluating classic ML models

**You will learn:** `mlflow.models.evaluate` for classifiers and regressors, custom
metrics, and automated validation against thresholds and a baseline.

## What `evaluate` does

Given a model URI (or a function, or a column of precomputed predictions) and labeled
data, the default evaluator:

* computes the standard metrics for the `model_type` (`classifier`, `regressor`, ...),
* logs plots as artifacts: confusion matrix, ROC and PR curves, lift and calibration,
* logs everything to the active run, and returns an `EvaluationResult`.

Add your own metrics with `make_metric(eval_fn=..., greater_is_better=...)`. The example
adds `missed_cancer_rate`: in this task a false negative costs far more than a false
positive, and accuracy hides that.

## Validation

```python
thresholds = {
  "f1_score": MetricThreshold(threshold=0.9, min_absolute_change=0.05, greater_is_better=True),
  "missed_cancer_rate": MetricThreshold(threshold=0.1, greater_is_better=False),
}
mlflow.validate_evaluation_results(thresholds, candidate_result, baseline_result)
```

This raises `ModelValidationFailedException` with a readable reason. Always compare against
a trivial baseline (`DummyClassifier`): it tells you what "no skill" scores on your data.

## Run it

`./book.sh run 11`. Open run `candidate` and view the plots under **Artifacts**.

## Exercises

1. Add a `regressor` evaluation for the diabetes model from chapter 10.
2. Use `evaluator_config={"pos_label": 0}` so that "malignant" is the positive class.
