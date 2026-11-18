"""Model registry: register versions, then promote through aliases behind a test-set gate."""
import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.metrics import f1_score

from common import CLASSIC_EXPERIMENT, REGISTERED_MODEL, TRACKING_URI, load_splits

CHAMPION, CHALLENGER = "champion", "challenger"


def test_f1(uri, X_test, y_test):
    return f1_score(y_test, mlflow.pyfunc.load_model(uri).predict(X_test))


def alias_version(client, alias):
    try:
        return client.get_model_version_by_alias(REGISTERED_MODEL, alias).version
    except MlflowException:
        return None


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    experiment = mlflow.get_experiment_by_name(CLASSIC_EXPERIMENT)
    _, _, X_test, _, _, y_test = load_splits()

    ranked = mlflow.search_logged_models(
        experiment_ids=[experiment.experiment_id],
        filter_string="metrics.val_f1 > 0",
        order_by=[{"field_name": "metrics.val_f1", "ascending": False}],
        output_format="list",
    )
    best, worst = ranked[0], ranked[-1]

    # On a clean registry, seed a weak "production" model so there is something to beat.
    if alias_version(client, CHAMPION) is None:
        seeded = mlflow.register_model(worst.model_uri, REGISTERED_MODEL)
        client.set_registered_model_alias(REGISTERED_MODEL, CHAMPION, seeded.version)
        client.update_model_version(
            REGISTERED_MODEL, seeded.version, description="Seeded baseline for the demo."
        )
        print(f"Seeded v{seeded.version} as @{CHAMPION}")

    candidate = mlflow.register_model(best.model_uri, REGISTERED_MODEL)
    client.set_registered_model_alias(REGISTERED_MODEL, CHALLENGER, candidate.version)
    print(f"Registered v{candidate.version} as @{CHALLENGER}")

    champion_f1 = test_f1(f"models:/{REGISTERED_MODEL}@{CHAMPION}", X_test, y_test)
    challenger_f1 = test_f1(f"models:/{REGISTERED_MODEL}@{CHALLENGER}", X_test, y_test)
    client.set_model_version_tag(
        REGISTERED_MODEL, candidate.version, "test_f1", f"{challenger_f1:.4f}"
    )
    print(f"Test F1: champion={champion_f1:.4f} challenger={challenger_f1:.4f}")

    if challenger_f1 > champion_f1:
        previous = alias_version(client, CHAMPION)
        client.set_registered_model_alias(REGISTERED_MODEL, CHAMPION, candidate.version)
        client.set_model_version_tag(
            REGISTERED_MODEL, candidate.version, "previous_champion", previous
        )
        print(f"PROMOTED v{candidate.version} to @{CHAMPION} (rollback target: v{previous})")
    else:
        client.set_model_version_tag(REGISTERED_MODEL, candidate.version, "gate", "failed")
        print(f"NOT promoted: @{CHAMPION} stays on v{alias_version(client, CHAMPION)}")


if __name__ == "__main__":
    main()
