# 13 · Hyperparameter tuning with Optuna

**You will learn:** how to structure a tuning study in MLflow so every trial is comparable
and the winner is easy to find.

## Pattern

```
parent run "optuna-gbr"     best_* params, best_cv_rmse, test_rmse, best_model, trials table
 ├─ trial-00  (nested)      params, cv_rmse, cv_rmse_std
 ├─ trial-01
 └─ ...
```

* Optuna proposes params with a sampler (TPE here) and decides what to try next.
* MLflow records each trial as a **nested run**, so the UI's parallel-coordinates and
  scatter plots work across trials.
* Refit the best params on the full training set and evaluate on the test set **once**.

## Key code

```python
with mlflow.start_run(run_name="optuna-gbr"):
    def objective(trial):
        params = {"max_depth": trial.suggest_int("max_depth", 1, 5), ...}
        with mlflow.start_run(nested=True):
            mlflow.log_params(params); ...; mlflow.log_metric("cv_rmse", rmse)
        return rmse
    optuna.create_study(direction="minimize").optimize(objective, n_trials=20)
```

## Run it

`./book.sh run 13`. In the UI, select the child runs and choose **Compare**, then the
**Parallel coordinates** plot.

## Pitfalls

* Tuning on the test set is leakage. Use cross-validation or a validation split.
* Run 20 trials with a fixed seed so results reproduce. Distributed studies need Optuna storage
  (RDB) as well as MLflow.
* Thousands of trials: log only the summary metrics per trial, and use `log_batch` or async logging.

## Exercises

1. Add Optuna pruning with `trial.report` and `trial.should_prune`, and log pruned trials with a tag.
2. Switch to `RandomForestRegressor` and compare the studies in one chart.
