"""Promotion gate for CI: compare @challenger with @champion on a holdout set.

Exit code 0 means promoted (or nothing to beat), 1 means the gate failed. A CI job can
run this after training and stop the pipeline on failure.

    python gate.py --model book-gated-classifier --min-f1 0.9 --min-gain 0.0
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.metrics import f1_score

from shared import TRACKING_URI, classification_splits


def version_for(client, name, alias):
    try:
        return client.get_model_version_by_alias(name, alias)
    except MlflowException:
        return None


def score(name, alias, X, y):
    return f1_score(y, mlflow.pyfunc.load_model(f"models:/{name}@{alias}").predict(X))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--min-f1", type=float, default=0.9)
    parser.add_argument("--min-gain", type=float, default=0.0)
    args = parser.parse_args()

    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    _, _, X_test, _, _, y_test = classification_splits()

    challenger = version_for(client, args.model, "challenger")
    if challenger is None:
        print("no @challenger to evaluate")
        return 1
    champion = version_for(client, args.model, "champion")
    new_f1 = score(args.model, "challenger", X_test, y_test)
    old_f1 = score(args.model, "champion", X_test, y_test) if champion else 0.0

    checks = {
        f"f1 >= {args.min_f1}": new_f1 >= args.min_f1,
        f"gain over champion > {args.min_gain}": new_f1 - old_f1 > args.min_gain,
    }
    passed = all(checks.values())
    client.set_model_version_tag(args.model, challenger.version, "gate_f1", f"{new_f1:.4f}")
    client.set_model_version_tag(args.model, challenger.version, "gate", "passed" if passed else "failed")
    if passed:
        client.set_registered_model_alias(args.model, "champion", challenger.version)
        if champion:
            client.set_model_version_tag(args.model, challenger.version, "rollback_to", champion.version)

    lines = [
        f"### Gate for `{args.model}` v{challenger.version}",
        f"challenger f1 = {new_f1:.4f}, champion f1 = {old_f1:.4f}"
        + (f" (v{champion.version})" if champion else " (none)"),
        *[f"- {'PASS' if ok else 'FAIL'}: {name}" for name, ok in checks.items()],
        f"**{'PROMOTED to @champion' if passed else 'NOT promoted'}**",
    ]
    print("\n".join(lines))
    # On GitHub Actions this shows up on the run's summary page.
    if os.getenv("GITHUB_STEP_SUMMARY"):
        Path(os.environ["GITHUB_STEP_SUMMARY"]).write_text("\n".join(lines) + "\n")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
