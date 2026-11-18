# 23 · Running a tracking server (local, Docker, production)

**You will learn:** server options, the Docker Compose stack in this repo, the REST API,
and what changes for a team deployment.

## Three setups

| Setup | Command | Backend | Artifacts |
|---|---|---|---|
| Book server | `./book.sh start` | SQLite `book/.data/mlflow.db` | `book/.data/artifacts`, proxied |
| Docker stack | `./book.sh docker-up` | Postgres 16 container | named volume, proxied |
| Team / production | see below | managed Postgres or MySQL | S3, GCS or Azure Blob, proxied |

```bash
mlflow server --host 0.0.0.0 --port 5000 \
  --backend-store-uri postgresql://user:pass@db:5432/mlflow \
  --artifacts-destination s3://my-bucket/mlflow \
  --allowed-hosts "mlflow.internal.example.com" \
  --workers 4
```

* `--artifacts-destination` with proxying (the default) means only the server holds
  bucket credentials. `--no-serve-artifacts` makes clients write to the bucket directly instead.
* `--allowed-hosts` and `--cors-allowed-origins` protect against DNS rebinding and CSRF.
* After upgrading MLflow, run `mlflow db upgrade <uri>` (chapter 25).

## The Docker stack (`docker-compose.yml`)

```
postgres:16-alpine  ◀── mlflow (ghcr.io/mlflow/mlflow:v3.18.0 + psycopg)  ──▶ :5050 on localhost
                              ▲
book (profile "book") ────────┘   MLFLOW_TRACKING_URI=http://mlflow:5000, repo copied to /work
```

```bash
./book.sh docker-up                  # postgres + mlflow, UI at http://127.0.0.1:5050
./book.sh docker-run 02 10 16        # run chapters in a container against that server
MLFLOW_TRACKING_URI=http://127.0.0.1:5050 ./book.sh run 23    # or from the host
./book.sh docker-down
```

Under WSL with Docker Desktop on Windows, `book.sh` calls `docker.exe` directly. Open
http://127.0.0.1:5050 in a Windows browser.

### Adding S3-compatible storage (untested sketch)

To use MinIO or another S3-compatible store, add a service and change the server flags:

```yaml
  minio:
    image: minio/minio
    command: server /data --console-address ":9001"
    environment: {MINIO_ROOT_USER: minio, MINIO_ROOT_PASSWORD: minio_demo_pw}
  mlflow:
    environment:
      MLFLOW_S3_ENDPOINT_URL: http://minio:9000
      AWS_ACCESS_KEY_ID: minio
      AWS_SECRET_ACCESS_KEY: minio_demo_pw
    command: >
      mlflow server ... --artifacts-destination s3://mlflow/
```

Create the bucket first, and add `boto3` to `docker/Dockerfile`. This was not run as part
of the book: anonymous MinIO image pulls were refused on this machine.

## REST API

Every client goes through `/api/2.0/mlflow/...`, for example `experiments/search`,
`runs/create`, `runs/log-metric` and `registered-models/get`. The example creates a run
using raw HTTP. That is handy from languages without an MLflow client (there are official
Java and R clients, and a TypeScript tracing SDK).

## Run it

`./book.sh run 23` (book server) or
`MLFLOW_TRACKING_URI=http://127.0.0.1:5050 ./book.sh run 23` (Docker stack).

## Production checklist

* Managed database with backups. Artifact bucket with lifecycle rules.
* Authentication: basic-auth (chapter 24) or SSO at a reverse proxy, plus TLS.
* Several server replicas behind a load balancer. The server is stateless.
* Monitoring of `/health`, DB connections and bucket errors. Retention: `mlflow gc` on a schedule.
