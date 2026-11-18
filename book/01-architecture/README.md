# 01 · What MLflow is and how it is built

**You will learn:** what each MLflow component is for, where data is stored, and how the
tracking URI decides everything else.

## The components

| Component | Answers the question | Chapters |
|---|---|---|
| Tracking | What did I try, with what data, and what happened? | 02–06 |
| Models | How do I package a model so anything can load it? | 07–09 |
| Model Registry | Which version is approved, and which one is live? | 10, 15 |
| Evaluation | Is this model or app good enough? | 11, 19 |
| Serving | How do I call the model over HTTP or in batch? | 12, 21 |
| Projects | How does someone else rerun my code? | 14 |
| Tracing | What happened inside one LLM or agent request? | 16, 17 |
| Prompt Registry | Which prompt version is in production? | 18 |
| Assessments | What did users, reviewers and judges think? | 20 |

## Two stores behind one server

```
 your code ──(REST)──▶ mlflow server ──▶ backend store   (SQL: runs, params, metrics, registry, traces)
                                    └──▶ artifact store  (files: models, plots, datasets)
```

* The **backend store** is a database: SQLite for one person, Postgres or MySQL for a team.
* The **artifact store** is a folder or a bucket: local disk, S3, GCS, Azure Blob.
* With artifact proxying (`--artifacts-destination`, the default for `mlflow server`), clients
  upload files *through* the server. Artifact URIs look like `mlflow-artifacts:/...`, and
  clients need no cloud credentials.

## The tracking URI

Everything follows from `mlflow.set_tracking_uri(...)` or the `MLFLOW_TRACKING_URI`
environment variable:

| URI | Meaning |
|---|---|
| `http://host:port` | A tracking server. The normal team setup. The book uses `http://127.0.0.1:5100`. |
| `sqlite:///path.db` | Talk to a database directly, with no server. Artifacts are written by the client. |
| `databricks` | Managed MLflow on Databricks. |

The registry URI defaults to the tracking URI. You can split them with `mlflow.set_registry_uri`.

## Run it

```bash
./book.sh start
./book.sh run 01
```

Open http://localhost:5100 and find the experiment `book-01-architecture`.

## Pitfalls

* With no tracking URI set and no server, MLflow writes to a local store in the current
  directory. Runs then "disappear" when you change directories. Always set the URI.
* SQLite is fine for one user. Concurrent writers need Postgres or MySQL (chapter 23).

## Exercises

1. Run the example with `MLFLOW_TRACKING_URI=sqlite:///scratch.db` and open the result with
   `mlflow ui --backend-store-uri sqlite:///scratch.db`.
2. Find the run's files on disk under `book/.data/artifacts`.
