"""Chapter 03: autologging.

One line captures params, training metrics, the fitted model and, for a grid search,
one child run per candidate. Metric calls you make after fitting are logged too.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV

from shared import banner, classification_splits, setup


def show(run_id: str) -> None:
    run = mlflow.get_run(run_id)
    params = dict(sorted(run.data.params.items())[:6])
    print(f"params (first 6 of {len(run.data.params)}): {params}")
    print(f"metrics: { {k: round(v, 4) for k, v in run.data.metrics.items()} }")
    print(f"tags   : { {k: v for k, v in run.data.tags.items() if k.startswith('estimator')} }")


def main():
    setup("book-03-autologging")
    X_train, X_val, _, y_train, y_val, _ = classification_splits()

    banner("Plain fit with sklearn autolog")
    # log_input_examples / log_model_signatures make the logged model self-describing.
    mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True, silent=True)
    with mlflow.start_run(run_name="autolog-gbm") as run:
        model = GradientBoostingClassifier(n_estimators=50, random_state=0).fit(X_train, y_train)
        # "Post-training metrics": calling a sklearn metric on this model's predictions is
        # logged automatically, named after the function and the dataset variable.
        f1_score(y_val, model.predict(X_val))
    show(run.info.run_id)
    models = mlflow.search_logged_models(
        filter_string=f"source_run_id = '{run.info.run_id}'", output_format="list"
    )
    print(f"logged models: {[(m.name, m.model_id) for m in models]}")

    banner("GridSearchCV creates a parent run plus child runs")
    search = GridSearchCV(
        GradientBoostingClassifier(random_state=0),
        {"n_estimators": [25, 50], "max_depth": [2, 3]},
        cv=3,
        scoring="f1",
    )
    with mlflow.start_run(run_name="autolog-gridsearch") as parent:
        search.fit(X_train, y_train)
    children = mlflow.search_runs(
        filter_string=f"tags.mlflow.parentRunId = '{parent.info.run_id}'", output_format="list"
    )
    print(f"best params: {search.best_params_}")
    print(f"child runs : {len(children)} (max_tuning_runs caps this, default 5)")

    banner("Turn it off, or keep metrics but skip models")
    # Calling autolog again replaces the previous settings for that library.
    # mlflow.autolog(...) does the same for every installed library at once.
    mlflow.sklearn.autolog(log_models=False, silent=True)
    with mlflow.start_run(run_name="autolog-no-model") as run:
        GradientBoostingClassifier(n_estimators=10, random_state=0).fit(X_train, y_train)
    models = mlflow.search_logged_models(
        filter_string=f"source_run_id = '{run.info.run_id}'", output_format="list"
    )
    print(f"params logged: {len(mlflow.get_run(run.info.run_id).data.params)}, models logged: {len(models)}")
    mlflow.sklearn.autolog(disable=True)


if __name__ == "__main__":
    main()
