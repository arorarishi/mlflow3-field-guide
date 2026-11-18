# 07 · The MLflow Model format: flavors, signatures, environments

**You will learn:** what is inside a logged model, why `pyfunc` matters, and how
signatures protect serving.

## Anatomy of a model folder

```
model/
  MLmodel              # flavors, signature, input example, mlflow version, model_id
  model.skops          # the serialized model (format depends on the flavor)
  input_example.json   serving_input_example.json
  requirements.txt     python_env.yaml   conda.yaml
```

* **Flavor:** the framework-specific way to save and load (`sklearn`, `pytorch`,
  `transformers`, ...). `mlflow.<flavor>.load_model` returns the *native* object.
* **python_function (pyfunc):** a universal flavor that every model also gets.
  `mlflow.pyfunc.load_model(uri).predict(df)` works the same for any framework. Serving,
  batch scoring and evaluation are built on it.
* **Signature:** the input and output schema, and optional inference params. It is enforced on
  every pyfunc `predict`: missing columns fail, and unsafe type conversions fail.
* **Input example:** documentation, the source of the signature, and a smoke test for serving.
* **Environment:** pinned requirements, inferred from what was imported, plus your extras.

## Model URIs

`runs:/<run_id>/model` · `models:/<model_id>` (MLflow 3) · `models:/<name>/<version>` ·
`models:/<name>@<alias>` · local paths · `s3://...`

## Key APIs

```python
sig = infer_signature(X_train, model.predict(X_train))
info = mlflow.sklearn.log_model(model, name="model", signature=sig, input_example=X.head(2),
                                extra_pip_requirements=["skops"])
mlflow.pyfunc.load_model(info.model_uri).predict(df)
mlflow.models.validate_serving_input(uri, payload)   # test a REST payload offline
mlflow.models.get_model_info(uri)                    # metadata without loading
```

## Run it

`./book.sh run 07`. The example prints the MLmodel file. Compare it with the UI's
**Artifacts** view of the model.

## Security note

Pickle-based formats can run code when they are loaded. Recent MLflow saves scikit-learn
models with **skops**, which refuses unknown types unless you list them in
`skops_trusted_types`. You will see that argument in chapters 09, 10 and 13. Only load
models from sources you trust.

## Exercises

1. Log the same model without a signature. What does `predict` accept now?
2. Use `mlflow models update-pip-requirements` to add a package to a logged model.
