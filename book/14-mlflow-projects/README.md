# 14 · MLflow Projects

**You will learn:** how to make code reproducible with one command, with named entry
points, typed parameters and a declared environment.

## The MLproject file

```yaml
name: book-diabetes-project
python_env: python_env.yaml          # or conda_env: / docker_env:
entry_points:
  main:
    parameters:
      alpha: {type: float, default: 1.0}
      l1_ratio: {type: float, default: 0.5}
    command: "python train.py --alpha {alpha} --l1-ratio {l1_ratio}"
```

`mlflow run <folder or git URL> -e main -P alpha=0.1` validates the parameters, builds the
environment (or uses the current one with `--env-manager local`), creates a run, and
runs the command with `MLFLOW_RUN_ID` set. The run records `mlflow.source.*` and
`mlflow.project.*` tags, including the git commit for git URLs.

```bash
mlflow run https://github.com/mlflow/mlflow-example -P alpha=0.5     # straight from git
mlflow run . -e report -P run_id=<id> --env-manager local
```

From Python: `mlflow.projects.run(uri, entry_point, parameters, env_manager, experiment_name)`.

## Where Projects fit today

Projects are useful for handing a reproducible job to a teammate or CI runner, and they
have backends for Databricks and Kubernetes. For multi-step pipelines, use an orchestrator
(Airflow, Prefect, Dagster, Kubeflow) that calls `mlflow run` or plain scripts.

## Run it

`./book.sh run 14`. Note that `alpha=1.0` gives r2 ≈ 0: the parameter is wrong for the
scale of this data. That kind of result is what tracking is for.

## Exercises

1. Remove `--env-manager local` and let MLflow build the virtualenv. It needs `pyenv` or `uv`.
2. Add a `tune` entry point that runs chapter 13's study.
