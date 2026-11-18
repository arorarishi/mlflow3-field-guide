"""Chapter 08: custom Python models.

1. A PythonModel subclass that wraps a trained model plus business logic and a config
   artifact. load_context() runs once at load time; predict() runs per request.
2. "Models from code": log a .py file instead of pickling an object (preferred for GenAI
   apps and anything that holds clients or connections).
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import mlflow
import pandas as pd
from mlflow.models import infer_signature
from sklearn.linear_model import Ridge

from shared import banner, regression_splits, setup

HERE = Path(__file__).parent


class RiskBandModel(mlflow.pyfunc.PythonModel):
    """Regression score plus a business rule that maps the score to a band."""

    def load_context(self, context):
        self.model = joblib.load(context.artifacts["regressor"])
        self.bands = json.loads(Path(context.artifacts["bands"]).read_text())

    def predict(self, context, model_input: pd.DataFrame, params=None):
        score = self.model.predict(model_input)
        band = pd.cut(score, bins=[-1e9, *self.bands["edges"], 1e9], labels=self.bands["labels"])
        return pd.DataFrame({"score": score.round(1), "band": band.astype(str)})


def main():
    setup("book-08-custom-pyfunc")
    X_train, X_test, y_train, _ = regression_splits()
    regressor = Ridge(alpha=0.5).fit(X_train, y_train)

    banner("1. PythonModel with artifacts")
    with tempfile.TemporaryDirectory() as tmp:
        joblib.dump(regressor, f"{tmp}/regressor.joblib")
        Path(f"{tmp}/bands.json").write_text(json.dumps({"edges": [100, 200], "labels": ["low", "medium", "high"]}))
        wrapper = RiskBandModel()
        # Build the signature from a real prediction so the output columns are exact.
        wrapper.model, wrapper.bands = regressor, json.loads(Path(f"{tmp}/bands.json").read_text())
        sample_out = wrapper.predict(None, X_test.head(5))
        with mlflow.start_run(run_name="risk-band"):
            info = mlflow.pyfunc.log_model(
                name="risk_band",
                python_model=RiskBandModel(),
                artifacts={"regressor": f"{tmp}/regressor.joblib", "bands": f"{tmp}/bands.json"},
                signature=infer_signature(X_test.head(5), sample_out),
                input_example=X_test.head(2),
                pip_requirements=["scikit-learn", "pandas", "joblib"],
            )
    loaded = mlflow.pyfunc.load_model(info.model_uri)
    print(loaded.predict(X_test.head(5)).to_string(index=False))

    banner("2. Models from code (type-hinted)")
    with mlflow.start_run(run_name="text-normalizer"):
        code_info = mlflow.pyfunc.log_model(
            name="normalizer",
            python_model=str(HERE / "model_code.py"),
            # A (data, params) tuple also records the params schema, so params are accepted.
            input_example=(["  Hello   World  "], {"lower": True}),
        )
    normalizer = mlflow.pyfunc.load_model(code_info.model_uri)
    print(normalizer.predict(["  Hello   World  ", "MLflow\tRocks"]))
    print(normalizer.predict(["Keep CASE"], params={"lower": False}))
    print(f"signature: inputs={normalizer.metadata.signature.inputs} params={normalizer.metadata.signature.params}")
    try:
        normalizer.predict([1, 2])
    except Exception as error:  # noqa: BLE001
        print(f"type hints are enforced: {type(error).__name__}: {str(error)[:90]}")


if __name__ == "__main__":
    main()
