# 10 · The Model Registry

**You will learn:** versions, aliases, tags and descriptions, and the release and
rollback workflow built on them.

## Concepts

* **Registered model:** a name, such as `book-diabetes-regressor`.
* **Version:** an immutable, numbered snapshot that points at one logged model.
* **Alias:** a movable named pointer (`@champion`, `@challenger`, `@shadow`). Each alias
  points at one version, and a version can have several aliases.
* **Tags and descriptions:** review state, validation results, owners, documentation.

Applications load `models:/name@champion`. Releasing is `set_registered_model_alias`.
Rolling back is the same call with the previous version. No application code changes.

> Stages (`Staging` / `Production`) are deprecated. They were a fixed list and allowed only
> one stage per version. Use aliases plus tags.

## Key APIs

```python
mlflow.sklearn.log_model(m, name="model", registered_model_name=NAME)   # log + register
mlflow.register_model(model_uri, NAME)                                  # register an existing model
client.set_registered_model_alias(NAME, "champion", version)
client.get_model_version_by_alias(NAME, "champion")
client.set_model_version_tag(NAME, v, "validation_status", "approved")
client.copy_model_version("models:/dev-name@champion", "prod-name")    # promote across names or registries
mlflow.search_model_versions("name = '...'")
```

## Run it

`./book.sh run 10`, then open **Models** in the UI.

## Patterns

* **Environment separation:** one registered model per environment (`dev`, `prod`), or
  separate registries. Promote with `copy_model_version` after the gate passes.
* **Who may move `@champion`?** Restrict it with permissions (chapter 24) and let CI do it
  (chapter 15).
* **Deployments do not follow aliases automatically.** A server resolves the alias when it
  starts. Restart or redeploy after moving it.

## Exercises

1. Add a `@shadow` alias and score both `@champion` and `@shadow` on the same data.
2. Write a `rollback.py` that reads the `rollback_to` tag (written in chapter 15) and moves the alias.
