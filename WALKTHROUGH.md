# 10-minute MLflow demo walkthrough

This is a guided path through seven book chapters. It covers tuning, registry, promotion
gate, serving, tracing, prompts and evaluation. Every step is a chapter you can read in
depth later. The numbers below come from real runs on MLflow 3.18.0 with the default
offline `fake` LLM.

## The 30-second version

> MLflow is an open-source platform for managing the ML and GenAI lifecycle. It records what
> you tried, versions what you decided to ship, and gives you a standard way to load and serve
> it. For LLM apps it adds tracing, prompt versioning and evaluation. It is a system of record.
> It does not orchestrate pipelines, store features or monitor drift by itself.

## How the pieces fit

| Concept | What it is | Where it appears |
|---|---|---|
| Experiment | A named group of related runs | `book-13-tuning`, `book-19-genai-eval` |
| Run | One execution: params, metrics, artifacts | Each Optuna trial (ch 13) |
| LoggedModel | First-class model entity in MLflow 3, with its own metrics | `models:/m-...` (ch 09, 13) |
| Flavor / pyfunc | Framework-specific format plus a universal `predict()` interface | ch 07, 12 |
| Signature | Input and output schema, enforced at serving | HTTP 400 in step 4 |
| Registered model | Named model with numbered versions | `book-diabetes-regressor` (ch 10) |
| Alias | A movable pointer to a version | `@champion`, `@challenger` |
| Trace / span | Tree of steps for one request, with inputs, outputs, timing and tokens | ch 16, 17 |
| Prompt registry | Immutable prompt versions with aliases | `prompts:/book-support-answer@production` |
| Scorer / judge | Quality check written in code, or run by an LLM | `has_expected_answer` (ch 19) |

Storage has two halves, and interviewers often ask about this (chapter 01):

- The **backend store** is a SQL database that holds metadata: runs, params, metrics, registry.
- The **artifact store** is blob storage that holds files: models, plots.
- With `--serve-artifacts` (the default), the server proxies artifact traffic, so clients need
  no storage credentials.

## Before you present

```bash
./book.sh start      # http://localhost:5100
./book.sh tour       # runs chapters 13 10 15 12 16 18 19 (about 2 minutes)
```

Open http://localhost:5100 and click through every screen once before you present. For a
clean slate, run `./book.sh stop && ./book.sh reset && ./book.sh start`, then run the tour
again.

## The demo (about 10 minutes)

### 1. Tracking: "Every experiment is recorded" ([ch 13](book/13-hyperparameter-tuning/))

Open experiment `book-13-tuning` and expand the `optuna-gbr` parent run.

> "I ran 20 Optuna trials of a gradient-boosting regressor as nested runs under one parent.
> Each trial records its params and cross-validated RMSE, so I can compare them all."

Sort the child runs by `cv_rmse`. The best trial used `max_depth=1` and a learning rate of about
0.08. It reached a CV RMSE of 56.83 and a test RMSE of 51.99. The top five trials all used
depth 1.

> "Shallow trees won. That is exactly why you track every trial instead of assuming that
> deeper models win."

Point at `start_run(nested=True)` in `example.py` and at the LoggedModel the parent run
produced.

### 2. Registry: "Releases are pointers" ([ch 10](book/10-model-registry/))

Open Models, then `book-diabetes-regressor`.

> "Two candidates were registered as versions: a ridge model with R² 0.419 and a forest
> with 0.469. `@champion` points to the forest and `@challenger` points to the ridge.
> Consumers load `models:/book-diabetes-regressor@champion`, so a release or a rollback
> moves a pointer and needs no code change."

The chapter shows a rollback (move `@champion` back and forth), a registry search, and a copy
of the champion into `book-diabetes-regressor-prod`. That copy is how you promote across
environments.

> "Stages like Staging and Production are deprecated in favor of aliases."

### 3. Promotion gate: "Promotion is a gated decision" ([ch 15](book/15-ci-promotion-gate/))

Open the registered model `book-gated-classifier`. On a fresh server, the gate sees three
candidates:

| Candidate | Test F1 | Gate result |
|---|---|---|
| dummy (always predicts the majority class) | 0.774 | fails the absolute bar of 0.9 |
| logreg C=1.0 | 0.986 | passes and becomes `@champion` |
| logreg C=0.001 | 0.917 | passes the bar but does not beat the champion |

> "`gate.py` exits non-zero when a candidate fails, so CI can act on the result. The
> chapter includes a GitHub Actions workflow that does this."

If you run the tour again, the strong model only ties the champion and is not promoted.
That shows the gate is real.

### 4. Serving: "Any model, same REST contract" ([ch 12](book/12-serving-deployment/))

The chapter starts `mlflow models serve` on port 5101 for
`models:/book-serving-classifier@champion` and calls it three ways:

```
POST /invocations (dataframe_split)  -> 200 {"predictions": [1, 0, 0]}   true labels: [1, 0, 0]
POST /invocations (missing column)   -> 400 INVALID_PARAMETER_VALUE
```

> "The serving layer does not care that this is scikit-learn, because it goes through pyfunc.
> The signature I logged at training time rejects a bad payload before it reaches the model."

The chapter also does batch scoring with `mlflow models predict` and generates a Dockerfile
with `mlflow models generate-dockerfile`.

### 5. Tracing: "See inside a request" ([ch 16](book/16-tracing-fundamentals/), [ch 17](book/17-tracing-llm-autolog/))

Open experiment `book-16-tracing`, then Traces, and open the trace with status ERROR:

```
order_status_bot   AGENT    ERROR
  extract_order_id   PARSER   OK
  policy_check       CHAIN    OK
  lookup_order       TOOL     ERROR
```

> "`@mlflow.trace` turns my own functions into spans. Failures are still logged, and the
> exception is attached to the span that raised it. The `api_key` input is masked before
> it is stored."

Filter by `trace.status = 'ERROR'` or by user. With a real LLM, chapter 17 adds
`mlflow.openai.autolog()`, which captures the chat and embedding calls in one line:
`LLM_PROVIDER=ollama ./book.sh run 17`.

### 6. Prompts and evaluation: "Prompt changes need evidence" ([ch 18](book/18-prompt-registry/), [ch 19](book/19-genai-evaluation/))

Open the Prompts page, then `book-support-answer`, and compare the versions. Then open
experiment `book-19-genai-eval` and the two evaluation runs.

| Prompt | Has expected answer | No markdown | Length OK |
|---|---|---|---|
| v1, loose | 75% | 100% | 100% |
| v2, grounded | 100% | 100% | 100% |

> "A prompt is a deployable artifact, so it gets versions and an alias like a model does. I ran
> both versions over the same 8-case dataset stored in MLflow. v2 won, so `@production`
> moved to it. The app loads the alias, so a prompt release needs no code deployment."

Open a failing v1 row to show detail for that answer. Be upfront about the limitation:
eight cases make a demo, not a benchmark. With `LLM_PROVIDER=ollama`, chapter 19 also adds
LLM judges.

### 7. Production shape: "This is how I would host it" ([ch 23](book/23-tracking-server-docker/), [ch 24](book/24-authentication/))

```bash
./book.sh docker-up          # http://127.0.0.1:5050, then ./book.sh docker-down
./book.sh docker-run 02      # run any chapter against it
```

> "Locally I used SQLite. For a team, that becomes Postgres for concurrent writes, object
> storage for artifacts, and authentication in front. This Compose file is the first step.
> Chapter 24 adds basic-auth with users and permissions."

## Questions you are likely to get

**Why use aliases instead of stages?** Stages were a fixed list, and a version could only be in
one stage at a time. Aliases are arbitrary named pointers, and a version can have several.
Tags carry any extra state.

**What makes a run reproducible?** Code version, params, data reference, environment and seed.
MLflow records the git commit when you run from a repo, logs pinned requirements with the
model, and tracks dataset metadata (ch 04). It does not snapshot your data, so the source must
be immutable or versioned.

**After moving an alias, does the running server change?** Not automatically.
`mlflow models serve` resolves the alias when it starts and loads that model. You restart or
roll the deployment. The alias is the source of truth, and your deployment system acts on it.

**How would you deploy this at scale?** `mlflow models build-docker` makes an image for
Kubernetes or a managed endpoint. There is also a SageMaker integration. CI moves the alias
after the gates pass (ch 15), then triggers a rolling or canary release.

**How do you secure it?** The server has no authentication by default. Use
`--app-name basic-auth` at minimum (ch 24), or preferably SSO through a reverse proxy. Add TLS
and private networking. `--allowed-hosts` guards against bad Host headers. Unpickling a model
can run code, so only load models from trusted sources. MLflow 3.18 saves scikit-learn models
with skops by default, which is why the chapters pass `skops_trusted_types`.

**Autolog or manual logging?** Autolog gives broad coverage quickly (ch 03). Add manual
logging for business metrics, custom plots and anything specific to the decision you need to
make.

**What does MLflow not do?** Pipeline orchestration (Airflow, Kubeflow), feature stores
(Feast), data versioning (DVC, lakeFS), and classic drift monitoring (Evidently, or your
observability stack).

**How does it compare to Weights & Biases?** W&B is a polished SaaS with strong collaboration
and visualization features. MLflow is open source and self-hostable, and its model packaging
and registry are broad. Databricks offers a managed MLflow. Compliance and hosting needs
usually decide between them.

**How does tracing work in production?** Export is asynchronous, so it adds little request
latency. Traces are OpenTelemetry-compatible. Use the lightweight `mlflow-tracing` package in
services, and sample high-volume traffic. Mask sensitive inputs before they are stored
(ch 16).

**Code scorers or LLM judges?** Use code for anything you can check in code, because it is
cheap and deterministic. Use judges for subjective qualities like groundedness or tone, and
validate judges against human labels (ch 20). Small local models make unreliable judges, so
keep them out of a promotion gate.

**What are the performance limits?** Logging many metrics one at a time is slow, so use
`log_batch` or asynchronous logging (ch 06). Set retention rules, because deleted runs are
soft-deleted until `mlflow gc` removes them (ch 25).

## Be straight about what this is

Describe it as "a project I built to get hands-on with MLflow 3", not as production
experience. You can say this with confidence, because you ran it:

- Every step runs end to end on public scikit-learn datasets (breast cancer, diabetes).
- The GenAI steps use a deterministic fake LLM unless you set a real provider.
- The promotion gate uses test F1. Gating repeatedly on the same test set slowly leaks
  information from it.
- The evaluation set has 8 cases, with simple substring checks.
- The Docker stack uses a local volume. Swapping it for real S3 was not tested here.

## If something breaks

| Symptom | Fix |
|---|---|
| `The book server is not running` | `./book.sh start` (log: `logs/book-server.log`) |
| Connection error with `LLM_PROVIDER=ollama` | Start Ollama. Under WSL, run `./book.sh ollama` and check the Windows firewall. |
| Port 5101 busy in chapter 12 | A previous model server is still running: `pkill -f "models serve"` |
| `docker-up` fails | Start Docker first. |
| You want a hosted model | Copy `.env.example` to `.env`, then set `LLM_PROVIDER` and the key. |
