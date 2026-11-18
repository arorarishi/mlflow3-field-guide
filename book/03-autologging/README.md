# 03 · Autologging

**You will learn:** what `autolog()` captures, how it behaves with grid searches, and how
to control it.

## What sklearn autolog records

* **Params:** every `get_params()` value of the estimator.
* **Training metrics:** accuracy, precision, recall, F1, log loss and ROC AUC on the training data.
* **Post-training metrics:** any `sklearn.metrics` call you make after `fit` on that model's
  predictions, named after the dataset variable (for example `f1_score_X_val`).
* **Model:** a LoggedModel with signature and input example if you enable them.
* **Search:** for `GridSearchCV` and `RandomizedSearchCV`, a parent run plus child runs
  (up to `max_tuning_runs`, default 5) and `best_*` params on the parent.
* **Tags:** `estimator_name`, `estimator_class`.

Other integrations work the same way: `mlflow.xgboost`, `lightgbm`, `pytorch` (Lightning),
`tensorflow`/`keras`, `transformers`, `spark`, and for GenAI `openai`, `langchain`,
`anthropic`, `llama_index`, `dspy` and others (chapter 17).

## Key APIs

```python
mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True, silent=True)
mlflow.sklearn.autolog(log_models=False)      # metrics and params only
mlflow.sklearn.autolog(disable=True)
mlflow.autolog()                              # every installed library
```

## Run it

`./book.sh run 03`. In the UI, expand `autolog-gridsearch` to see the child runs.

## Autolog or manual logging?

Autolog gives broad coverage with almost no code. Add manual logging for business metrics,
custom plots and the exact evaluation that drives your decision. They mix freely in one run.

## Pitfalls

* Autolog patches library functions at call time. Call it **before** you create and fit models.
* It logs *training-set* metrics. Do not report those as model quality.
* In notebooks, re-running cells can create unexpected extra runs. Wrap work in `start_run`.

## Exercises

1. Add `exclusive=True` and log a manual metric. What happens?
2. Use `RandomizedSearchCV` with 20 candidates and set `max_tuning_runs=20`.
