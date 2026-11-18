# 21 · Packaging an agent with ResponsesAgent

**You will learn:** how to give an agent a standard interface so that it can be logged,
registered, evaluated and served like a model.

## Why an interface

`ResponsesAgent` is a `PythonModel` with an OpenAI Responses-style schema:

```json
request:  {"input": [{"role": "user", "content": "..."}], "context": {...}, "custom_inputs": {...}}
response: {"output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "..."}]}]}
```

* The signature is set for you, so there is no schema work.
* `predict_stream` yields `response.output_text.delta` events, and serving streams them.
* Tool calls and reasoning items have helpers: `create_function_call_item`,
  `create_function_call_output_item` and `create_reasoning_item`.
* Calls are traced automatically.

## Logging from code

```python
mlflow.pyfunc.log_model(name="agent", python_model="agent_code.py",
                        code_paths=["shared.py", "support_bot.py"],
                        input_example={"input": [{"role": "user", "content": "..."}]},
                        registered_model_name="book-support-agent")
```

`agent_code.py` ends with `mlflow.models.set_model(SupportAgent())`. Any framework works
inside `predict`: LangGraph, OpenAI Agents SDK, DSPy or plain Python.

## Run it

`./book.sh run 21`. Serve it with
`mlflow models serve -m models:/book-support-agent/latest -p 5101 --env-manager local` and
POST the same JSON to `/invocations`.

## Exercises

1. Return a `function_call` item for the `get_price` tool from chapter 17.
2. Evaluate the registered agent with chapter 19's dataset: `predict_fn=mlflow.genai.to_predict_fn(uri)`.
