# 18 · The Prompt Registry

**You will learn:** how to version prompts, release them with aliases, and keep model
settings with the prompt.

## Concepts

* `register_prompt(name, template)` creates a new **immutable version**, with a commit
  message and tags, much like a git commit.
* Templates use `{{variable}}` placeholders. `prompt.format(**vars)` fills them in.
  `template` can be a string or a list of chat messages.
* **Aliases** (`@production`, `@staging`) are movable pointers. `@latest` is always the newest version.
* `model_config` stores the model name, temperature and other settings with the version.
* The UI shows versions side by side with a **diff**.

## Key APIs

```python
v = mlflow.genai.register_prompt(name, template, commit_message="...", model_config={...})
mlflow.genai.set_prompt_alias(name, alias="production", version=v.version)
p = mlflow.genai.load_prompt("prompts:/name@production")       # or /3, or @latest
p.format(context=..., question=...)
mlflow.genai.search_prompts("name LIKE 'book-%'")
```

## Release flow

1. Register a new version, and point `@staging` at it.
2. Evaluate staging against production on the same dataset (chapter 19).
3. If it wins, move `@production`. The app loads by alias and picks it up on its next load.
   No deployment is needed.

## Run it

`./book.sh run 18`. Every run registers new versions, so watch the version numbers grow.

## Exercises

1. Register a prompt with `response_format` (a Pydantic model) for structured output.
2. Cache `load_prompt` in your app for 60 seconds, so an alias move takes effect without a restart.
