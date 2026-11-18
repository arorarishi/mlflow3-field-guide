"""Entry point `report`: reads another run and logs a short summary as its own run."""
import argparse

import mlflow

parser = argparse.ArgumentParser()
parser.add_argument("--run-id", required=True)
args = parser.parse_args()

source = mlflow.get_run(args.run_id)
with mlflow.start_run():
    mlflow.set_tag("source_run", args.run_id)
    mlflow.log_text(f"r2 of {args.run_id}: {source.data.metrics['r2']:.3f}\n", "report.txt")
    print(f"report written for {args.run_id}")
