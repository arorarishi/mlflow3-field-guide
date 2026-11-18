"""Chapter 07: the MLflow Model format.

A logged model is a folder with an MLmodel file. It lists the *flavors* the model can be
loaded as (here `sklearn` and the universal `python_function`), the signature, and the
environment. The signature is enforced whenever the model is loaded through pyfunc.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow
import numpy as np
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from mlflow.types import ColSpec, ParamSchema, ParamSpec, Schema
from mlflow.models.signature import ModelSignature
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from shared import banner, classification_splits, setup


def short(error: Exception) -> str:
    """Schema errors embed the whole payload. Keep only the reason at the end."""
    return str(error).rsplit("Error:", 1)[-1].strip()[:160]


def main():
    setup("book-07-models")
    X_train, X_val, _, y_train, y_val, _ = classification_splits()
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X_train, y_train)

    banner("Log with an inferred signature and an input example")
    signature = infer_signature(X_train, model.predict(X_train))
    with mlflow.start_run(run_name="logreg"):
        info = mlflow.sklearn.log_model(
            model,
            name="model",
            signature=signature,
            input_example=X_train.head(2),
            # Extra lines appended to the inferred requirements.txt:
            extra_pip_requirements=["skops"],
        )
    print(f"model_uri : {info.model_uri}")
    print(f"flavors   : {list(info.flavors)}")
    print(f"signature : {len(signature.inputs.inputs)} inputs -> {signature.outputs.to_dict()}")

    banner("What is inside the model folder")
    local = Path(mlflow.artifacts.download_artifacts(info.model_uri))
    print(sorted(p.name for p in local.iterdir()))
    print((local / "MLmodel").read_text())
    print("requirements.txt:", (local / "requirements.txt").read_text().split())

    banner("Load it two ways")
    native = mlflow.sklearn.load_model(info.model_uri)  # the real sklearn Pipeline
    generic = mlflow.pyfunc.load_model(info.model_uri)  # same predict() for every flavor
    print(f"native  : {type(native).__name__}, predict_proba available: {hasattr(native, 'predict_proba')}")
    print(f"pyfunc  : {type(generic).__name__}, predictions {generic.predict(X_val.head(5)).tolist()}")

    banner("Schema enforcement")
    try:
        generic.predict(X_val.head(2).drop(columns=["mean radius"]))
    except MlflowException as error:
        print(f"missing column rejected: {short(error)}")
    as_strings = X_val.head(2).astype(str)
    try:
        generic.predict(as_strings)
    except MlflowException as error:
        print(f"wrong type rejected   : {short(error)}")

    banner("Check a REST payload offline before deploying")
    payload = mlflow.models.convert_input_example_to_serving_input(X_val.head(1))
    print(f"serving payload starts: {payload[:80]}...")
    print(f"validate_serving_input -> {mlflow.models.validate_serving_input(info.model_uri, payload)}")

    banner("Signatures can also declare inference params")
    manual = ModelSignature(
        inputs=Schema([ColSpec("double", "x")]),
        outputs=Schema([ColSpec("double")]),
        params=ParamSchema([ParamSpec("threshold", "double", 0.5)]),
    )
    print(manual.to_dict()["params"])
    print("pyfunc.predict(data, params={'threshold': 0.7}) passes them to the model.")
    _ = np  # numpy arrays are valid inputs as well: infer_signature(np_array, preds)


if __name__ == "__main__":
    main()
