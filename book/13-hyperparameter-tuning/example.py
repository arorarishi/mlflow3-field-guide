"""Chapter 13: hyperparameter tuning with Optuna.

Optuna chooses the next trial and MLflow records every trial. Each trial is a nested run
under one parent, so the UI can compare all trials and the parent holds the winner.
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow
import optuna
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

from shared import SEED, banner, regression_splits, setup

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
N_TRIALS = 20


def main():
    setup("book-13-tuning")
    X_train, X_test, y_train, y_test = regression_splits()

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 20, 200, step=20),
            "max_depth": trial.suggest_int("max_depth", 1, 5),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        }
        with mlflow.start_run(run_name=f"trial-{trial.number:02d}", nested=True):
            mlflow.log_params(params)
            model = GradientBoostingRegressor(random_state=SEED, **params)
            scores = cross_val_score(model, X_train, y_train, cv=3, scoring="neg_root_mean_squared_error")
            rmse = -scores.mean()
            mlflow.log_metrics({"cv_rmse": rmse, "cv_rmse_std": scores.std()})
        return rmse

    banner(f"Run {N_TRIALS} Optuna trials as nested MLflow runs")
    with mlflow.start_run(run_name="optuna-gbr") as parent:
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
        study.optimize(objective, n_trials=N_TRIALS)

        # Refit the winner on all training data and record it on the parent run.
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best_cv_rmse", study.best_value)
        best = GradientBoostingRegressor(random_state=SEED, **study.best_params).fit(X_train, y_train)
        test_rmse = mean_squared_error(y_test, best.predict(X_test)) ** 0.5
        mlflow.log_metric("test_rmse", test_rmse)
        info = mlflow.sklearn.log_model(best, name="best_model", input_example=X_train.head(2),
                                        skops_trusted_types=["sklearn.tree._tree.Tree", "sklearn._loss.loss.HalfSquaredError"])
        # Optuna's own plots are plotly figures. Log the trial table instead (no extra deps).
        mlflow.log_table(study.trials_dataframe(), "optuna/trials.json")

    print(f"best params : {study.best_params}")
    print(f"best cv rmse: {study.best_value:.2f}   test rmse: {test_rmse:.2f}")
    print(f"model       : {info.model_uri}")

    banner("Top 5 trials straight from MLflow")
    top = mlflow.search_runs(
        filter_string=f"tags.mlflow.parentRunId = '{parent.info.run_id}'",
        order_by=["metrics.cv_rmse ASC"],
        max_results=5,
    )
    print(top[["tags.mlflow.runName", "params.max_depth", "params.learning_rate", "metrics.cv_rmse"]].to_string(index=False))


if __name__ == "__main__":
    main()
