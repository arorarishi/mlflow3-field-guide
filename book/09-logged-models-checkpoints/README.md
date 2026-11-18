# 09 · LoggedModel: models as first-class entities (MLflow 3)

**You will learn:** the MLflow 3 model entity, how to attach metrics to a specific model,
and how to pick the best checkpoint.

## Concepts

In MLflow 2, a model was a folder of artifacts inside a run. In MLflow 3, every
`log_model` creates a **LoggedModel** with its own ID (`m-...`), name, params, tags,
metrics and status. It links back to the run that produced it, but it can be searched on
its own. One run can produce many models: checkpoints, candidates, quantized variants.

```
Run "mlp-30-epochs"
 ├─ checkpoint-05  m-52..  metrics: val_loss@step5,  val_accuracy@step5
 ├─ checkpoint-10  m-ed..  ...
 └─ checkpoint-30  m-6f..  ...
```

## Key APIs

```python
info = mlflow.sklearn.log_model(model, name="checkpoint-10", step=10)
mlflow.log_metrics({...}, step=10, model_id=info.model_id, dataset=val_ds)
mlflow.search_logged_models(experiment_ids=[...], filter_string="metrics.val_loss < 0.1",
                            order_by=[{"field_name": "metrics.val_loss", "ascending": True}])
mlflow.set_logged_model_tags(model_id, {...});  mlflow.log_model_params({...}, model_id=...)
mlflow.create_external_model(name="vendor-api")   # track something MLflow did not package
mlflow.set_active_model(name="app-v2")            # link traces to a model (chapter 22)
```

## Run it

`./book.sh run 09`. In the UI, open the experiment's **Models** tab. Each checkpoint is a
row with its own metrics.

## Exercises

1. Raise the learning rate until later checkpoints overfit, so the best checkpoint is not the last one.
2. Filter LoggedModels by `tags.selected = 'true'`.
