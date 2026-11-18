"""Chapter 10: the Model Registry.

Registered models have numbered, immutable versions. Aliases (@champion) are movable
pointers that consumers load, so a release or a rollback is a pointer move, not a code
change. Tags and descriptions carry review state and documentation.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from shared import banner, regression_splits, setup

NAME = "book-diabetes-regressor"
PROD_NAME = "book-diabetes-regressor-prod"
TRUSTED = ["sklearn.tree._tree.Tree"]


def train_and_register(model, run_name, X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name=run_name):
        model.fit(X_train, y_train)
        r2 = r2_score(y_test, model.predict(X_test))
        mlflow.log_metric("test_r2", r2)
        # registered_model_name logs the model and creates a new version in one call.
        info = mlflow.sklearn.log_model(
            model,
            name="model",
            input_example=X_train.head(2),
            registered_model_name=NAME,
            skops_trusted_types=TRUSTED,
        )
    print(f"{run_name:8s} r2={r2:.3f} -> {NAME} v{info.registered_model_version}")
    return info.registered_model_version, r2


def main():
    setup("book-10-registry")
    client = MlflowClient()
    X_train, X_test, y_train, y_test = regression_splits()

    banner("Register two versions")
    v_ridge, r2_ridge = train_and_register(Ridge(alpha=1.0), "ridge", X_train, X_test, y_train, y_test)
    v_forest, r2_forest = train_and_register(
        RandomForestRegressor(n_estimators=100, max_depth=4, random_state=0), "forest", X_train, X_test, y_train, y_test
    )

    banner("Document the model and its versions")
    client.update_registered_model(NAME, description="Predicts diabetes progression one year out.")
    client.set_registered_model_tag(NAME, "owner", "book-reader")
    client.update_model_version(NAME, v_ridge, description="Linear baseline.")
    client.set_model_version_tag(NAME, v_forest, "validation_status", "approved")

    banner("Aliases: champion and challenger")
    champion, challenger = (v_ridge, v_forest) if r2_ridge >= r2_forest else (v_forest, v_ridge)
    client.set_registered_model_alias(NAME, "champion", champion)
    client.set_registered_model_alias(NAME, "challenger", challenger)
    for alias in ("champion", "challenger"):
        version = client.get_model_version_by_alias(NAME, alias)
        print(f"@{alias:10s} -> v{version.version} (run {version.run_id[:8]})")

    banner("Consumers load by alias")
    model = mlflow.pyfunc.load_model(f"models:/{NAME}@champion")
    print(f"models:/{NAME}@champion predicts {model.predict(X_test.head(3)).round(1).tolist()}")

    banner("Rollback = move the pointer back")
    client.set_registered_model_alias(NAME, "champion", challenger)
    print(f"champion now v{client.get_model_version_by_alias(NAME, 'champion').version}")
    client.set_registered_model_alias(NAME, "champion", champion)
    print(f"rolled back to v{champion}")

    banner("Search the registry")
    for found in mlflow.search_model_versions(filter_string=f"name = '{NAME}'", order_by=["version_number DESC"]):
        # Search results are summaries. Fetch the version for aliases.
        version = client.get_model_version(NAME, found.version)
        print(f"v{version.version} aliases={version.aliases} tags={version.tags}")
    print([m.name for m in mlflow.search_registered_models(filter_string="tags.owner = 'book-reader'")])

    banner("Promote across environments by copying a version")
    # A common pattern: a 'dev' model and a 'prod' model, possibly in different registries.
    copied = client.copy_model_version(f"models:/{NAME}@champion", PROD_NAME)
    client.set_registered_model_alias(PROD_NAME, "live", copied.version)
    print(f"{PROD_NAME} v{copied.version} @live (copied from {NAME}@champion)")

    banner("Deleting an alias")
    client.delete_registered_model_alias(NAME, "challenger")
    try:
        client.get_model_version_by_alias(NAME, "challenger")
    except MlflowException:
        print("@challenger no longer resolves. Loading it would fail loudly, which is what you want.")


if __name__ == "__main__":
    main()
