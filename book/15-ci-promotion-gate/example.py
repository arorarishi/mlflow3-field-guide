"""Chapter 15: a promotion gate you can run in CI.

Trains candidates, tags each as @challenger and runs gate.py, which promotes it only when
it clears an absolute bar *and* beats the current @champion. The gate's exit code is what
a CI system acts on.

    python example.py               full demo: three candidates through the gate
    python example.py --train-only  just train one candidate (what CI would do)
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow
from mlflow import MlflowClient
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from shared import banner, classification_splits, cli_env, setup

NAME = "book-gated-classifier"
GATE = Path(__file__).parent / "gate.py"


def train_challenger(label, model):
    X_train, _, _, y_train, _, _ = classification_splits()
    with mlflow.start_run(run_name=label):
        model.fit(X_train, y_train)
        info = mlflow.sklearn.log_model(model, name="model", input_example=X_train.head(2), registered_model_name=NAME)
    MlflowClient().set_registered_model_alias(NAME, "challenger", info.registered_model_version)
    print(f"{label}: v{info.registered_model_version} is @challenger")


def gate():
    result = subprocess.run([sys.executable, str(GATE), "--model", NAME, "--min-f1", "0.9"], env=cli_env())
    print(f"gate exit code: {result.returncode}")


def main():
    setup("book-15-ci-gate")
    if "--train-only" in sys.argv:
        train_challenger("ci-candidate", make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000)))
        return

    banner("1. A useless model never clears the absolute bar")
    train_challenger("dummy", DummyClassifier(strategy="most_frequent"))
    gate()

    banner("2. A strong model clears it and becomes @champion")
    train_challenger("logreg-C1", make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000)))
    gate()

    banner("3. A weaker model loses to the champion")
    train_challenger("logreg-C0.001", make_pipeline(StandardScaler(), LogisticRegression(C=0.001, max_iter=1000)))
    gate()


if __name__ == "__main__":
    main()
