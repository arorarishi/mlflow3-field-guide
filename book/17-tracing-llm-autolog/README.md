# 17 · Tracing LLM calls with autolog

**You will learn:** one-line tracing for LLM SDKs, and how to combine it with your own
spans to see a whole agent loop.

> Needs a real model. With Ollama: `LLM_PROVIDER=ollama ./book.sh run 17`. With the fake
> provider, this chapter prints a message and exits.

## Autolog integrations

`mlflow.openai.autolog()`, `mlflow.anthropic.autolog()`, `mlflow.langchain.autolog()`
(which also covers LangGraph), `mlflow.llama_index.autolog()`, `mlflow.dspy.autolog()`,
`mlflow.bedrock.autolog()`, `mlflow.gemini.autolog()`, `mlflow.litellm.autolog()`, and more.
Each one creates spans for model calls with messages, parameters, **token usage** and latency,
and handles streaming.

Because Ollama, DeepInfra, vLLM and many others expose an OpenAI-compatible API, the OpenAI
integration traces all of them. Changing provider is configuration, not code
(`book/shared.py` → `PROVIDERS`).

## Composition

```
pricing_agent  AGENT        (your @mlflow.trace)
 ├─ Completions CHAT_MODEL  (autolog) — model decides to call get_price
 ├─ get_price   TOOL        (your @mlflow.trace)
 └─ Completions CHAT_MODEL  (autolog) — final answer
```

Do not wrap autologged calls in extra spans. Add spans for *your* steps: routing,
retrieval, tools and post-processing.

## Run it

```bash
./demo.sh ollama &                     # WSL: starts Windows Ollama on the WSL gateway
LLM_PROVIDER=ollama ./book.sh run 17
```

## What to look for

* Token counts per call, and the total on the trace (`trace.info.token_usage`).
* With small models such as `llama3.2:1b`, the agent often sends malformed tool arguments.
  The example returns the error to the model as a tool result, and the trace shows where it
  went wrong. Debugging that case is what tracing is for.

## Exercises

1. Add a second tool and ask a question that needs both.
2. Search for traces whose total tokens exceed 200.
