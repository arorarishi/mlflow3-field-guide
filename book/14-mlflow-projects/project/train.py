"""Entry point `main`. `mlflow run` passes params as CLI flags and sets MLFLOW_RUN_ID,
so start_run() below attaches to the run that `mlflow run` already created."""
import argparse

import mlflow
from sklearn.datasets import load_diabetes
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, required=True)
parser.add_argument("--l1-ratio", type=float, required=True)
args = parser.parse_args()

X, y = load_diabetes(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

with mlflow.start_run():
    model = ElasticNet(alpha=args.alpha, l1_ratio=args.l1_ratio).fit(X_train, y_train)
    r2 = r2_score(y_test, model.predict(X_test))
    mlflow.log_metric("r2", r2)
    mlflow.sklearn.log_model(model, name="model", input_example=X_train.head(2))
    print(f"alpha={args.alpha} l1_ratio={args.l1_ratio} r2={r2:.3f}")
