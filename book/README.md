# The MLflow Book

A hands-on course in MLflow 3, from the first logged metric to authenticated servers and
evaluated agents. Every chapter has a `README.md` (concepts, APIs, pitfalls and exercises)
and an `example.py` that runs as written. All 25 examples were run against MLflow 3.18.0.

## Quick start

```bash
./setup.sh                 # Python 3.12 venv in .venv (uses uv, or falls back to pip)
./book.sh start            # tracking server on http://localhost:5100
./book.sh run 02           # one chapter. Or: ./book.sh run 07 08 09, or ./book.sh all
./book.sh stop
```

Docker instead of a local server: `./book.sh docker-up` then `./book.sh docker-run 02`
(chapter 23).

GenAI chapters run **offline** with a deterministic fake LLM by default. For real model
calls, set `LLM_PROVIDER=ollama` (or `openai` / `deepinfra` with a key in `.env`).

## Contents

### Part I — Tracking
| # | Chapter | You will learn |
|---|---|---|
| 01 | [Architecture](01-architecture/) | components, backend vs artifact store, tracking URIs |
| 02 | [Tracking basics](02-tracking-basics/) | runs, params, metrics, tags, artifacts, resuming runs |
| 03 | [Autologging](03-autologging/) | sklearn autolog, grid-search child runs, controls |
| 04 | [Datasets and lineage](04-datasets-lineage/) | `mlflow.data`, digests, contexts |
| 05 | [Search and query](05-search-and-query/) | filter language, pagination, soft delete |
| 06 | [System metrics and async](06-system-metrics-async/) | CPU and memory metrics, batch and async logging |

### Part II — Models
| # | Chapter | You will learn |
|---|---|---|
| 07 | [Models, flavors, signatures](07-models-flavors-signatures/) | MLmodel format, pyfunc, schema enforcement |
| 08 | [Custom pyfunc](08-custom-pyfunc/) | PythonModel with artifacts, models-from-code, type hints |
| 09 | [LoggedModels and checkpoints](09-logged-models-checkpoints/) | MLflow 3 model entity, per-model metrics |
| 10 | [Model Registry](10-model-registry/) | versions, aliases, rollback, cross-environment copy |
| 11 | [Model evaluation](11-model-evaluation/) | `mlflow.models.evaluate`, custom metrics, validation |
| 12 | [Serving and deployment](12-serving-deployment/) | REST serving, batch CLI, Docker images |

### Part III — Workflow
| # | Chapter | You will learn |
|---|---|---|
| 13 | [Hyperparameter tuning](13-hyperparameter-tuning/) | Optuna with nested runs |
| 14 | [MLflow Projects](14-mlflow-projects/) | MLproject, entry points, `mlflow run` |
| 15 | [CI promotion gate](15-ci-promotion-gate/) | automated champion/challenger with exit codes |

### Part IV — GenAI
| # | Chapter | You will learn |
|---|---|---|
| 16 | [Tracing fundamentals](16-tracing-fundamentals/) | spans, decorators, redaction, trace search |
| 17 | [LLM tracing with autolog](17-tracing-llm-autolog/) | OpenAI-compatible autolog, tool-calling agent *(needs a real LLM)* |
| 18 | [Prompt Registry](18-prompt-registry/) | prompt versions, aliases, chat templates |
| 19 | [GenAI evaluation](19-genai-evaluation/) | datasets, code scorers, LLM judges |
| 20 | [Feedback and assessments](20-feedback-assessments/) | user and reviewer labels, overrides, regression sets |
| 21 | [Agents with ResponsesAgent](21-agents-responsesagent/) | packaging, streaming, serving an agent |
| 22 | [App versioning](22-app-versioning/) | `set_active_model`, comparing versions |

### Part V — Operations
| # | Chapter | You will learn |
|---|---|---|
| 23 | [Tracking server and Docker](23-tracking-server-docker/) | server flags, Compose stack, REST API |
| 24 | [Authentication](24-authentication/) | basic-auth, users, permissions |
| 25 | [Housekeeping](25-housekeeping/) | gc, migrations, export |

## Suggested paths

* **New to MLflow:** 01 → 02 → 03 → 05 → 07 → 10 → 12
* **ML engineer:** 04, 06, 08, 09, 11, 13, 15
* **GenAI / agents:** 16 → 17 → 18 → 19 → 20 → 21 → 22
* **Platform / MLOps:** 12, 14, 15, 23, 24, 25

## Layout

```
book/
  shared.py        tracking URI, datasets, LLM provider switch, fake LLM
  support_bot.py   small traced RAG bot used by chapters 19–22
  NN-topic/        README.md + example.py (+ supporting files)
  .data/           the book server's database and artifacts (./book.sh reset deletes it)
```

## Conventions

* Each chapter writes to its own experiment, `book-NN-...`, and registered models start with `book-`.
* Examples are re-runnable. Re-running adds new runs and versions instead of failing.
* Ports: 5100 book server, 5101 model serving (ch 12), 5102 auth server (ch 24), 5050 Docker stack.
