"""Chapter 23: talking to a tracking server.

Works against any server: the local book server (port 5100) or the Docker stack
(port 5050, Postgres + proxied artifacts). It checks health, uses the REST API directly
(what every client library does underneath), and round-trips an artifact through the
server's proxy, so clients need no storage credentials.

    MLFLOW_TRACKING_URI=http://127.0.0.1:5050 python example.py   # against the Docker stack
"""
import json
import sys
import tempfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow

from shared import TRACKING_URI, banner, run_cli, setup


def rest(method: str, path: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        f"{TRACKING_URI}/api/2.0/mlflow/{path}", data=data, method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read() or b"{}")


def main():
    banner(f"Server at {TRACKING_URI}")
    for path in ("/health", "/version"):
        with urllib.request.urlopen(TRACKING_URI + path, timeout=5) as response:
            print(f"GET {path:8s} -> {response.read().decode().strip()}")

    banner("The REST API underneath the Python client")
    experiment = setup("book-23-server")
    found = rest("POST", "experiments/search", {"max_results": 3, "order_by": ["creation_time DESC"]})
    print("latest experiments:", [e["name"] for e in found.get("experiments", [])])
    run = rest("POST", "runs/create", {"experiment_id": experiment.experiment_id, "run_name": "created-over-rest"})["run"]
    run_id = run["info"]["run_id"]
    rest("POST", "runs/log-metric", {"run_id": run_id, "key": "via_rest", "value": 1.0, "timestamp": 0, "step": 0})
    rest("POST", "runs/update", {"run_id": run_id, "status": "FINISHED"})
    print(f"created run {run_id} with raw HTTP. Python sees: {mlflow.get_run(run_id).data.metrics}")

    banner("Artifacts go through the server (proxied artifact store)")
    with mlflow.start_run(run_name="artifact-roundtrip") as active:
        mlflow.log_text("stored by the server, not by this client", "proof.txt")
    print(f"artifact_uri = {active.info.artifact_uri}")
    with tempfile.TemporaryDirectory() as tmp:
        run_cli("artifacts", "download", "--run-id", active.info.run_id, "--artifact-path", "proof.txt", "--dst-path", tmp,
                capture_output=True)
        print("downloaded:", (Path(tmp) / "proof.txt").read_text())
    print("mlflow-artifacts:/ URIs mean the client never needs S3 / GCS / Azure credentials.")


if __name__ == "__main__":
    main()
