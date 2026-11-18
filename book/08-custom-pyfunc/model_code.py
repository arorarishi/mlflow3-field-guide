"""Models from code: MLflow stores this *file*, not a pickle of the object.

Type hints on predict() become the model signature, and inputs are validated against them.
"""
import re

import mlflow
from mlflow.pyfunc import PythonModel


class TextNormalizer(PythonModel):
    def predict(self, model_input: list[str], params: dict | None = None) -> list[str]:
        lower = (params or {}).get("lower", True)
        cleaned = [re.sub(r"\s+", " ", text).strip() for text in model_input]
        return [text.lower() if lower else text for text in cleaned]


mlflow.models.set_model(TextNormalizer())
