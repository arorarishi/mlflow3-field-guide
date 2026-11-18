"""Chapter 14: MLflow Projects.

A project is a folder with an MLproject file that names entry points, their typed
parameters and the environment. `mlflow run` turns "how do I run this?" into one command,
for a local folder or a Git URL, and records the source and version on the run.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow

from shared import banner, cli_env, run_cli, setup

PROJECT = Path(__file__).parent / "project"
EXPERIMENT = "book-14-projects"


def main():
    setup(EXPERIMENT)
    # MLproject commands call `python`. Make sure that means this interpreter.
    os.environ.update(cli_env())

    banner("Run the 'main' entry point from Python, twice with different params")
    run_ids = []
    for alpha in (0.01, 1.0):
        submitted = mlflow.projects.run(
            uri=str(PROJECT),
            entry_point="main",
            parameters={"alpha": alpha, "l1_ratio": 0.5},
            env_manager="local",  # use this interpreter. Default builds python_env.yaml.
            experiment_name=EXPERIMENT,
        )
        run_ids.append(submitted.run_id)
        print(f"alpha={alpha}: run {submitted.run_id} status={submitted.get_status()}")

    banner("Run the 'report' entry point from the CLI")
    env = cli_env()
    env.pop("MLFLOW_EXPERIMENT_ID", None)  # set by the runs above. The CLI flag is clearer.
    run_cli("run", str(PROJECT), "-e", "report", "-P", f"run_id={run_ids[0]}",
            "--env-manager", "local", "--experiment-name", EXPERIMENT, env=env)

    banner("What MLflow recorded about the source")
    tags = mlflow.get_run(run_ids[0]).data.tags
    for key, value in sorted(tags.items()):
        if key.startswith(("mlflow.source", "mlflow.project")):
            print(f"{key:28s} = {value}")
    print("Bad params fail before anything runs. Try: mlflow run <dir> -P alpha=abc")


if __name__ == "__main__":
    main()
