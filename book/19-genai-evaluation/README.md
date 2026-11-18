# 19 · Evaluating GenAI apps

**You will learn:** evaluation datasets, code scorers, built-in and custom LLM judges, and
how to compare two versions of an app on evidence.

## The pieces

```python
mlflow.genai.evaluate(
    data=dataset,                # list of dicts, DataFrame, MLflow EvaluationDataset, or traces
    predict_fn=my_app,           # called with **inputs. Omit it to score existing outputs or traces
    scorers=[...],
)
```

Each row is `{"inputs": {...}, "expectations": {...}}`. Every `predict_fn` call is traced,
and every scorer result is attached to that trace as an assessment.

| Scorer | Example | Needs an LLM |
|---|---|---|
| Code `@scorer` | `has_expected_answer(outputs, expectations) -> bool` | no |
| Code scorer reading the trace | `used_retrieval(trace) -> Feedback` | no |
| Built-in deterministic | `RegexMatch`, `ResponseLength` | no |
| Built-in judges | `Correctness`, `Guidelines`, `RelevanceToQuery`, `RetrievalGroundedness`, `Safety`, ... | yes |
| Custom judge | `make_judge(name, instructions="... {{ outputs }} ...", model=...)` | yes |

**Evaluation datasets** (`mlflow.genai.datasets.create_dataset` / `merge_records`) are
versioned in MLflow and can grow from production traces (chapter 20).

## Judge models

Judges take a model URI such as `openai:/gpt-4o-mini`. The book points `OPENAI_BASE_URL` at
the configured provider, so `openai:/llama3.2:1b` uses Ollama (`shared.judge_model()`).
Small models produce judges you cannot trust. Before a judge gates anything, validate it
against human labels.

## Run it

```bash
./book.sh run 19                         # offline: code scorers only
LLM_PROVIDER=ollama ./book.sh run 19     # adds Correctness, Guidelines and a custom judge
```

Then open the two `eval-prompt-*` runs and use **Compare** in the evaluation view.

## Pitfalls

* Eight cases is a demo. Real suites need tens to hundreds of cases, covering refusals and edge cases.
* Writing the prompt while looking at the test cases leaks the test set into the prompt.
* Use code checks for anything checkable. They are cheap, deterministic and debuggable.

## Exercises

1. Add a `RetrievalSufficiency` judge and a case where retrieval misses the right document.
2. Use `mlflow.genai.evaluate(data=traces_df, scorers=[...])` to score last week's traces.
