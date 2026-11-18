# MLflow 3 Field Guide

A hands-on course in MLflow 3, from the first logged metric to authenticated servers and
evaluated agents. It has 25 chapters. Each one has a `README.md` covering concepts, APIs,
pitfalls and exercises, and an `example.py` that runs as written. All examples were run
against MLflow 3.18.0.

Every demo in this repo is a book chapter. They all use the same server, data and helpers.
The [10-minute walkthrough](WALKTHROUGH.md) is a guided path through seven of the chapters.

## Quick start

```bash
./setup.sh                 # Python 3.12 venv in .venv (uses uv, or falls back to pip)
./book.sh start            # tracking server on http://localhost:5100
./book.sh run 02           # one chapter. Or: ./book.sh run 07 08 09
./book.sh tour             # the 10-minute demo path: chapters 13 10 15 12 16 18 19
./book.sh all              # every chapter, with a pass/fail summary
./book.sh stop
```

Requirements: [uv](https://docs.astral.sh/uv/) (preferred) or Python 3.12. Docker and
Ollama are optional.

GenAI chapters run **offline** by default, using a deterministic fake LLM. For real model
calls, set `LLM_PROVIDER=ollama`, or use `openai` or `deepinfra` with a key in `.env` (see
`.env.example`).

To use Docker instead of a local server, run `./book.sh docker-up`, then
`./book.sh docker-run 02`. This gives you Postgres plus MLflow on port 5050 (chapter 23).

## Contents

### Part I: Tracking
| # | Chapter | You will learn |
|---|---|---|
| 01 | [Architecture](book/01-architecture/) | components, backend vs artifact store, tracking URIs |
| 02 | [Tracking basics](book/02-tracking-basics/) | runs, params, metrics, tags, artifacts, resuming runs |
| 03 | [Autologging](book/03-autologging/) | sklearn autolog, grid-search child runs, controls |
| 04 | [Datasets and lineage](book/04-datasets-lineage/) | `mlflow.data`, digests, contexts |
| 05 | [Search and query](book/05-search-and-query/) | filter language, pagination, soft delete |
| 06 | [System metrics and async](book/06-system-metrics-async/) | CPU and memory metrics, batch and async logging |

### Part II: Models
| # | Chapter | You will learn |
|---|---|---|
| 07 | [Models, flavors, signatures](book/07-models-flavors-signatures/) | MLmodel format, pyfunc, schema enforcement |
| 08 | [Custom pyfunc](book/08-custom-pyfunc/) | PythonModel with artifacts, models-from-code, type hints |
| 09 | [LoggedModels and checkpoints](book/09-logged-models-checkpoints/) | MLflow 3 model entity, per-model metrics |
| 10 | [Model Registry](book/10-model-registry/) | versions, aliases, rollback, cross-environment copy |
| 11 | [Model evaluation](book/11-model-evaluation/) | `mlflow.models.evaluate`, custom metrics, validation |
| 12 | [Serving and deployment](book/12-serving-deployment/) | REST serving, batch CLI, Docker images |

### Part III: Workflow
| # | Chapter | You will learn |
|---|---|---|
| 13 | [Hyperparameter tuning](book/13-hyperparameter-tuning/) | Optuna with nested runs |
| 14 | [MLflow Projects](book/14-mlflow-projects/) | MLproject, entry points, `mlflow run` |
| 15 | [CI promotion gate](book/15-ci-promotion-gate/) | automated champion/challenger with exit codes |

### Part IV: GenAI
| # | Chapter | You will learn |
|---|---|---|
| 16 | [Tracing fundamentals](book/16-tracing-fundamentals/) | spans, decorators, redaction, trace search |
| 17 | [LLM tracing with autolog](book/17-tracing-llm-autolog/) | OpenAI-compatible autolog, tool-calling agent *(needs a real LLM)* |
| 18 | [Prompt Registry](book/18-prompt-registry/) | prompt versions, aliases, chat templates |
| 19 | [GenAI evaluation](book/19-genai-evaluation/) | datasets, code scorers, LLM judges |
| 20 | [Feedback and assessments](book/20-feedback-assessments/) | user and reviewer labels, overrides, regression sets |
| 21 | [Agents with ResponsesAgent](book/21-agents-responsesagent/) | packaging, streaming, serving an agent |
| 22 | [App versioning](book/22-app-versioning/) | `set_active_model`, comparing versions |

### Part V: Operations
| # | Chapter | You will learn |
|---|---|---|
| 23 | [Tracking server and Docker](book/23-tracking-server-docker/) | server flags, Compose stack, REST API |
| 24 | [Authentication](book/24-authentication/) | basic-auth, users, permissions |
| 25 | [Housekeeping](book/25-housekeeping/) | gc, migrations, export |

The whole book is also available as a PDF: [book/MLflow-Book.pdf](book/MLflow-Book.pdf)
(rebuild it with `./book.sh pdf`).

## Suggested paths

* **10-minute demo:** `./book.sh tour`, then talk through it with [WALKTHROUGH.md](WALKTHROUGH.md)
* **New to MLflow:** 01 → 02 → 03 → 05 → 07 → 10 → 12
* **ML engineer:** 04, 06, 08, 09, 11, 13, 15
* **GenAI / agents:** 16 → 17 → 18 → 19 → 20 → 21 → 22
* **Platform / MLOps:** 12, 14, 15, 23, 24, 25

## Repository layout

| Path | Purpose |
|---|---|
| `book/NN-topic/` | one chapter: `README.md` + `example.py` (+ supporting files) |
| `book/shared.py` | tracking URI, datasets, LLM provider switch, fake LLM |
| `book/support_bot.py` | small traced RAG bot used by chapters 19–22 |
| `book.sh` | server, chapter runner, tour, Docker helpers |
| `setup.sh`, `requirements.txt` | create or verify the virtualenv |
| `docker-compose.yml`, `docker/` | Postgres + MLflow server, book runner image |
| `.env.example` | LLM provider and keys (copy to `.env`) |
| `WALKTHROUGH.md` | talk track and Q&A for the 10-minute demo |

## Conventions

* Each chapter writes to its own experiment, `book-NN-...`, and registered models start
  with `book-`.
* Examples can be re-run. Re-running adds new runs and versions instead of failing.
* Ports: 5100 book server, 5101 model serving (ch 12), 5102 auth server (ch 24),
  5050 Docker stack.
* `./book.sh reset` (with the server stopped) deletes everything under `book/.data/`.
