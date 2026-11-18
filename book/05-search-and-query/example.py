"""Chapter 05: finding runs again.

search_runs with filter strings, ordering, pagination with the client, experiment
search by tag, and the soft-delete lifecycle.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow
from mlflow import MlflowClient
from mlflow.entities import ViewType

from shared import banner, setup


def seed_runs(experiment_id: str) -> None:
    if mlflow.search_runs([experiment_id], max_results=1, output_format="list"):
        return
    for model in ("ridge", "lasso", "forest"):
        for alpha in (0.1, 1.0, 10.0):
            with mlflow.start_run(run_name=f"{model}-{alpha}"):
                mlflow.log_params({"model": model, "alpha": alpha})
                score = {"ridge": 0.80, "lasso": 0.75, "forest": 0.88}[model] - alpha / 100
                mlflow.log_metric("r2", score)
                mlflow.set_tag("team", "pricing" if model != "forest" else "growth")


def main():
    experiment = setup("book-05-search")
    client = MlflowClient()
    client.set_experiment_tag(experiment.experiment_id, "domain", "pricing")
    seed_runs(experiment.experiment_id)

    banner("Filter, order, select columns (pandas)")
    df = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="metrics.r2 > 0.78 AND params.model != 'lasso'",
        order_by=["metrics.r2 DESC"],
    )
    print(df[["tags.mlflow.runName", "params.model", "params.alpha", "metrics.r2"]].to_string(index=False))

    banner("More filter syntax")
    queries = [
        "tags.team = 'growth'",
        "attributes.run_name LIKE 'ridge%'",
        "params.alpha = '10.0'",
        "attributes.run_name ILIKE '%FOREST%'",
        "attributes.status = 'FINISHED' AND metrics.r2 >= 0.85",
    ]
    for query in queries:
        runs = mlflow.search_runs([experiment.experiment_id], filter_string=query, output_format="list")
        print(f"{query:55s} -> {len(runs)} runs")

    banner("Best run in one call")
    best = mlflow.search_runs(
        [experiment.experiment_id], order_by=["metrics.r2 DESC"], max_results=1, output_format="list"
    )[0]
    print(f"best: {best.info.run_name} r2={best.data.metrics['r2']:.3f}")

    banner("Paginate with the client (for large experiments)")
    token, pages = None, 0
    while True:
        page = client.search_runs([experiment.experiment_id], max_results=4, page_token=token)
        pages += 1
        token = page.token
        if not token:
            break
    print(f"read all runs in {pages} pages of 4")

    banner("Search experiments by tag")
    found = mlflow.search_experiments(filter_string="tags.domain = 'pricing'")
    print(f"experiments tagged domain=pricing: {[e.name for e in found]}")

    banner("Soft delete and restore")
    victim = best.info.run_id
    client.delete_run(victim)
    active = len(mlflow.search_runs([experiment.experiment_id], output_format="list"))
    deleted = len(
        mlflow.search_runs([experiment.experiment_id], run_view_type=ViewType.DELETED_ONLY, output_format="list")
    )
    print(f"after delete: active={active} deleted={deleted}")
    client.restore_run(victim)
    print("restored. A deleted run is only gone for good after `mlflow gc` (chapter 25).")


if __name__ == "__main__":
    main()
