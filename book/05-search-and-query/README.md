# 05 · Searching and querying runs

**You will learn:** the filter language, how to find the best run in code, and the run
lifecycle.

## Filter syntax

```
metrics.r2 > 0.8 AND params.model = 'ridge' AND tags.team != 'growth'
attributes.status = 'FINISHED'      attributes.run_name LIKE 'ridge%'   ILIKE is case-insensitive
attributes.start_time > 1700000000000                                    (milliseconds)
tags.`mlflow.parentRunId` = '<id>'                                       backticks for dotted keys
datasets.name = 'train'    datasets.digest = 'b83cd940'
```

* Only `AND` is supported. There is no `OR`. Run two searches and merge the results.
* Param and tag values are strings, so compare them with quotes: `params.alpha = '0.1'`.
* `order_by=["metrics.r2 DESC"]` and `max_results=1` give you "the best run".

## Key APIs

```python
mlflow.search_runs(experiment_ids, filter_string, order_by, max_results)         # DataFrame
mlflow.search_runs(..., output_format="list")                                    # Run objects
MlflowClient().search_runs(..., max_results=100, page_token=token)               # pagination
mlflow.search_experiments(filter_string="tags.domain = 'pricing'")
client.delete_run(id); client.restore_run(id); run_view_type=ViewType.DELETED_ONLY
```

## Run it

`./book.sh run 05`. Then try the same filters in the UI search box. The syntax is identical.

## Exercises

1. Find runs where `r2` is in the top 3 for each `params.model`. Use pandas `groupby`.
2. Use `mlflow runs list --experiment-id <id>` from the CLI.
