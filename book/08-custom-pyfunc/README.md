# 08 · Custom Python models

**You will learn:** how to package anything as an MLflow model: pre- and post-processing,
business rules, several models, or code with no ML in it at all.

## Two styles

**1. Subclass `mlflow.pyfunc.PythonModel` and pass an object.**

```python
class RiskBandModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):          # once, at load time
        self.model = joblib.load(context.artifacts["regressor"])
    def predict(self, context, model_input, params=None):
        ...
mlflow.pyfunc.log_model(name="m", python_model=RiskBandModel(),
                        artifacts={"regressor": "regressor.joblib"})
```

`artifacts` are files copied into the model. `load_context` receives their local paths.
The object is serialized with cloudpickle, which is why MLflow prints a warning.

**2. Models from code (recommended).** Log a `.py` file that ends with
`mlflow.models.set_model(MyModel())`. MLflow stores the source, not a pickle. This style is
easy to review and diff, avoids serialization problems with clients or locks, and is
the standard for GenAI apps and agents (chapter 21).

```python
mlflow.pyfunc.log_model(name="normalizer", python_model="model_code.py",
                        input_example=(["  Hello  "], {"lower": True}))
```

## Type-hint signatures

`def predict(self, model_input: list[str], params=None) -> list[str]` makes the type hints
the signature, and MLflow validates inputs against them. Pydantic models also work as hints.
Passing an input example as a `(data, params)` tuple adds a params schema.

## Run it

`./book.sh run 08`

## Pitfalls

* Imports of your own modules: pass `code_paths=[...]` or the loaded model breaks elsewhere.
* Set `pip_requirements` explicitly for wrappers. Inference cannot always see what
  `load_context` imports.

## Exercises

1. Add a `params={"threshold": ...}` option to `RiskBandModel` and record it in the signature.
2. Rewrite `RiskBandModel` as models-from-code.
