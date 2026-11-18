# 25 · Housekeeping: deletion, garbage collection, migrations, export

**You will learn:** how MLflow deletes data, how to reclaim space, and how to keep the
database healthy across upgrades.

## Soft delete, then garbage collection

* `delete_run` and `delete_experiment` only set `lifecycle_stage = deleted`. The data stays,
  is hidden from default searches, and can be restored.
* `mlflow gc` permanently removes deleted runs, their metrics and their **artifacts**:

```bash
mlflow gc --backend-store-uri postgresql://... --older-than 30d
mlflow gc --tracking-uri http://server:5000 --run-ids <id1>,<id2>    # through a server
```

Schedule it (cron or CI) as part of your retention policy. Registered model versions and
prompts have their own delete APIs.

## Upgrades

1. Back up the database.
2. Upgrade the `mlflow` package on the server.
3. `mlflow db upgrade <backend-uri>` (idempotent).
4. Restart the servers. Clients can usually be one minor version apart.

## Export and inspection

```bash
mlflow experiments csv -x <id> -o runs.csv
mlflow runs describe --run-id <id>
mlflow artifacts download --run-id <id> --dst-path ./out
mlflow doctor                       # versions and config for bug reports
```

To move whole experiments between servers, use the community `mlflow-export-import` tool.

## Run it

`./book.sh run 25`. It uses a throwaway local store, so your data is safe.

## Exercises

1. Write a script that deletes runs tagged `stage=exploration` older than 14 days, then runs `gc`.
2. Run `./book.sh reset` and `./book.sh all` to rebuild the whole book from scratch.
