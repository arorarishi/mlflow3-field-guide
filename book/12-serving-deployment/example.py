"""Chapter 12: serving and deployment.

`mlflow models serve` turns any pyfunc model into a REST service. This script registers a
model, starts the server as a subprocess, calls it with every supported payload format,
runs the batch-prediction CLI, and generates (but does not build) a Dockerfile.
"""
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow
from mlflow import MlflowClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from shared import banner, classification_splits, cli_env, mlflow_cli, run_cli, setup

NAME = "book-serving-classifier"
PORT = int(os.getenv("BOOK_SERVE_PORT", "5101"))
BASE = f"http://127.0.0.1:{PORT}"


def call(path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(BASE + path, data, {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, response.read().decode().strip()
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode()[:160]


def main():
    setup("book-12-serving")
    X_train, X_val, _, y_train, y_val, _ = classification_splits()
    with mlflow.start_run(run_name="to-serve"):
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X_train, y_train)
        info = mlflow.sklearn.log_model(model, name="model", input_example=X_train.head(2), registered_model_name=NAME)
    MlflowClient().set_registered_model_alias(NAME, "champion", info.registered_model_version)
    uri = f"models:/{NAME}@champion"

    banner(f"Start a REST server for {uri} on port {PORT}")
    env = cli_env()
    server = subprocess.Popen(
        [mlflow_cli(), "models", "serve", "-m", uri, "-p", str(PORT), "--env-manager", "local"],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(90):
            if server.poll() is not None:
                sys.exit(f"model server exited with code {server.returncode}")
            try:
                if call("/ping")[0] == 200:
                    break
            except OSError:
                pass
            time.sleep(1)
        print("GET /ping    ->", call("/ping"))
        print("GET /version ->", call("/version"))

        rows = X_val.head(3)
        payloads = {
            "dataframe_split": {"dataframe_split": rows.to_dict(orient="split")},
            "dataframe_records": {"dataframe_records": rows.to_dict(orient="records")},
            "inputs": {"inputs": rows.to_dict(orient="list")},
        }
        for label, payload in payloads.items():
            print(f"POST /invocations ({label:17s}) ->", call("/invocations", payload))
        print(f"true labels: {y_val.head(3).tolist()}")
        bad = {"dataframe_split": rows.drop(columns=["mean radius"]).to_dict(orient="split")}
        print("POST /invocations (missing column)  ->", call("/invocations", bad))
    finally:
        server.terminate()
        server.wait(timeout=30)

    with tempfile.TemporaryDirectory() as tmp:
        banner("Batch scoring from the CLI")
        X_val.head(4).to_csv(f"{tmp}/input.csv", index=False)
        run_cli("models", "predict", "-m", uri, "-i", f"{tmp}/input.csv", "-t", "csv",
                "-o", f"{tmp}/out.json", "--env-manager", "local", env=env, stderr=subprocess.DEVNULL)
        print("predictions:", Path(f"{tmp}/out.json").read_text().strip())

        banner("Generate a Dockerfile (build it with `docker build`)")
        run_cli("models", "generate-dockerfile", "-m", uri, "-d", f"{tmp}/image", "--env-manager", "local",
                env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(sorted(p.name for p in Path(f"{tmp}/image").iterdir()))
        print("\n".join(Path(f"{tmp}/image/Dockerfile").read_text().splitlines()[:12]))
        print("...\n`mlflow models build-docker -m <uri> -n <image>` does the generate and build in one step.")


if __name__ == "__main__":
    main()
