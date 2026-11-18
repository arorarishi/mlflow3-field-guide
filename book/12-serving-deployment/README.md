# 12 · Serving and deployment

**You will learn:** how to serve any MLflow model over REST, score in batch, and build a
container image.

## Local REST serving

```bash
mlflow models serve -m "models:/book-serving-classifier@champion" -p 5101 --env-manager local
```

| Endpoint | Purpose |
|---|---|
| `GET /ping`, `GET /health` | liveness |
| `GET /version` | MLflow version |
| `POST /invocations` | predict |

Payload formats for `/invocations`:

```json
{"dataframe_split": {"columns": [...], "data": [[...]]}}
{"dataframe_records": [{"col": 1.0, ...}]}
{"inputs": {"col": [1.0, 2.0]}}      or      {"instances": [...]}     (tensor-style)
{"inputs": ..., "params": {"threshold": 0.7}}
```

`--env-manager` decides the environment: `local` uses the current one. `virtualenv`, `uv`
and `conda` build the model's pinned environment, which is slower but matches training.

## Batch and Spark

```bash
mlflow models predict -m <uri> -i input.csv -t csv -o out.json --env-manager local
```

In Spark, use `mlflow.pyfunc.spark_udf(spark, uri)` to score a DataFrame in parallel.

## Containers

```bash
mlflow models generate-dockerfile -m <uri> -d ./image    # inspect, then docker build
mlflow models build-docker -m <uri> -n my-model:1         # generate and build
docker run -p 8080:8080 my-model:1
```

The image serves the same REST API on port 8080. It runs on Kubernetes, ECS, Cloud Run
or Azure ML. `mlflow sagemaker` and `mlflow deployments` plugins target managed platforms.

## Run it

`./book.sh run 12`. The example starts and stops its own server on port 5101.

## Production checklist

* Pin the version you deploy (`models:/name/7`), or resolve the alias at deploy time and record the version.
* Health checks on `/ping`. Readiness waits for the model to load.
* Log requests and responses with an ID, and keep latency and error dashboards.
* After moving an alias, roll the deployment. The server does not reload by itself.

## Exercises

1. Build the image with Docker and call it on port 8080.
2. Send 1,000 rows in one request and measure latency against batch prediction.
