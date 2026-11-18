"""Chapter 24: authentication and permissions with the built-in basic-auth app.

Starts a separate, throwaway MLflow server with `--app-name basic-auth` (port 5102,
temporary storage), then: creates users as admin, grants per-experiment permissions,
and shows what each user can and cannot do. Clients authenticate with the
MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD environment variables.
"""
import os
import secrets
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow
from mlflow.server import get_app_client
from mlflow.exceptions import MlflowException

from shared import banner, cli_env, mlflow_cli

PORT = int(os.getenv("BOOK_AUTH_PORT", "5102"))
URI = f"http://127.0.0.1:{PORT}"
ADMIN_PASSWORD = "book-admin-Passw0rd!"  # demo only. Use a secret store for real ones.
USERS = {"alice": "alice-Passw0rd-123!", "bob": "bob-Passw0rd-1234!"}


@contextmanager
def as_user(username: str, password: str):
    """Credentials are read from the environment on each request."""
    os.environ["MLFLOW_TRACKING_USERNAME"], os.environ["MLFLOW_TRACKING_PASSWORD"] = username, password
    try:
        yield
    finally:
        os.environ.pop("MLFLOW_TRACKING_USERNAME", None)
        os.environ.pop("MLFLOW_TRACKING_PASSWORD", None)


def attempt(label, action):
    try:
        result = action()
        print(f"  OK      {label}" + (f" -> {result}" if result else ""))
    except MlflowException as error:
        print(f"  DENIED  {label}: {error.error_code}")


def main():
    with tempfile.TemporaryDirectory() as tmp:
        config = Path(tmp) / "auth.ini"
        config.write_text(
            "[mlflow]\n"
            "default_permission = READ\n"
            f"database_uri = sqlite:///{tmp}/auth.db\n"
            "admin_username = admin\n"
            f"admin_password = {ADMIN_PASSWORD}\n"
            "authorization_function = mlflow.server.auth:authenticate_request_basic_auth\n"
        )
        banner(f"Start an auth-enabled server on {URI}")
        server = subprocess.Popen(
            [mlflow_cli(), "server", "--app-name", "basic-auth", "--host", "127.0.0.1", "--port", str(PORT),
             "--backend-store-uri", f"sqlite:///{tmp}/mlflow.db", "--artifacts-destination", f"{tmp}/artifacts"],
            # The auth app needs a static Flask secret for CSRF protection. Keep it identical
            # across replicas and out of source control in a real deployment.
            env=cli_env(MLFLOW_AUTH_CONFIG_PATH=str(config), MLFLOW_FLASK_SERVER_SECRET_KEY=secrets.token_hex(16)),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        try:
            for _ in range(90):
                try:
                    urllib.request.urlopen(f"{URI}/health", timeout=2)
                    break
                except OSError:
                    if server.poll() is not None:
                        sys.exit("auth server exited early")
                    time.sleep(1)
            mlflow.set_tracking_uri(URI)

            banner("No credentials")
            try:
                urllib.request.urlopen(f"{URI}/api/2.0/mlflow/experiments/search?max_results=1", timeout=5)
            except urllib.error.HTTPError as error:
                print(f"  GET experiments/search -> HTTP {error.code} {error.reason}")

            banner("Admin creates users and a private experiment")
            with as_user("admin", ADMIN_PASSWORD):
                auth = get_app_client("basic-auth", tracking_uri=URI)
                for name, password in USERS.items():
                    auth.create_user(username=name, password=password)
                    print(f"  created user {name}")
                experiment_id = mlflow.create_experiment("team-alpha")
                auth.grant_user_permission("alice", "experiment", experiment_id, "EDIT")
                print(f"  experiment team-alpha ({experiment_id}): alice=EDIT, everyone else=READ (default)")

            banner("alice (EDIT on team-alpha)")
            with as_user("alice", USERS["alice"]):
                attempt("log a run in team-alpha", lambda: mlflow.start_run(experiment_id=experiment_id).info.run_id[:8])
                mlflow.end_run()
                attempt("read team-alpha", lambda: mlflow.get_experiment(experiment_id).name)

            banner("bob (default READ)")
            with as_user("bob", USERS["bob"]):
                attempt("read team-alpha", lambda: mlflow.get_experiment(experiment_id).name)
                attempt("log a run in team-alpha", lambda: mlflow.start_run(experiment_id=experiment_id).info.run_id[:8])
                attempt("create his own experiment", lambda: mlflow.create_experiment("bob-sandbox"))
                print("  (creating an experiment makes bob its MANAGE owner)")

            banner("Wrong password")
            with as_user("bob", "not-his-password"):
                attempt("read team-alpha", lambda: mlflow.get_experiment(experiment_id).name)
        finally:
            server.terminate()
            server.wait(timeout=30)
    print("\nThe throwaway auth server is stopped and its data deleted.")


if __name__ == "__main__":
    main()
