# 22 · Versioning a GenAI application

**You will learn:** how to track app versions (code + prompt + model + config) and compare
them by their traces and evaluation results.

## Concepts

A GenAI app's behaviour depends on more than weights. MLflow 3 records an app version as a
**LoggedModel**, even if nothing is packaged:

```python
active = mlflow.set_active_model(name="support-bot-v2")      # create or reuse by name
mlflow.log_model_params({"prompt_version": "v2", "llm": "..."}, model_id=active.model_id)
# every trace from now on is linked to support-bot-v2
mlflow.genai.evaluate(..., model_id=active.model_id)         # eval metrics land on the version
mlflow.search_traces(model_id=active.model_id)
```

Good version params: git commit, prompt name and version, LLM name, temperature,
retriever settings. `mlflow.genai.enable_git_model_versioning()` can create versions
from git state automatically.

## Run it

`./book.sh run 22`. In the UI, open the experiment's **Models** tab (versions with their
eval metrics) and filter Traces by model.

## Exercises

1. Add a `v3` that retrieves 1 document instead of 2, and compare the three versions.
2. Record the git commit with `mlflow.genai.enable_git_model_versioning()`.
