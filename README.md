# MLflow 3 Field Guide

Two things live here:

| | What | Start with |
|---|---|---|
| **The MLflow Book** | 25 chapters covering all of MLflow 3, each with notes and a runnable example | [book/README.md](book/README.md) |
| **Interview demo** | a 10-minute end-to-end demo: tracking → registry → serving → tracing → prompt eval | [WALKTHROUGH.md](WALKTHROUGH.md) |

## Setup

```bash
./setup.sh             # creates .venv (Python 3.12, MLflow 3.18.0)
./setup.sh --check     # verify an existing install
```

Requirements: [uv](https://docs.astral.sh/uv/) (preferred) or Python 3.12. Optional:
Docker (Compose stack), Ollama (real LLM calls).

## Book

```bash
./book.sh start && ./book.sh run 01      # http://localhost:5100
./book.sh all                            # every chapter, with a pass/fail summary
./book.sh docker-up && ./book.sh docker-run 02    # Postgres-backed server in Docker
```

## Demo

```bash
./start.sh        # Docker Desktop, Ollama, demo server on :5000
./demo.sh all
./stop.sh
```

The book (port 5100, `book/.data/`) and the demo (port 5000, `mlflow.db`) use separate
servers and stores, so they never mix.

| File | Purpose |
|---|---|
| `setup.sh` | create or verify the virtualenv |
| `book.sh` | book server, chapter runner, Docker helpers |
| `requirements.txt` | Python dependencies |
| `docker-compose.yml`, `docker/` | Postgres + MLflow server, book runner image |
| `.env.example` | LLM provider and keys (copy to `.env`) |
| `demo.sh`, `start.sh`, `stop.sh`, `0*.py` | the interview demo |
