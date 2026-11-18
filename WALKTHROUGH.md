# MLflow demo walkthrough

Built and run on MLflow 3.18.0. All numbers below come from real runs of this demo.

## The 30-second version

> MLflow is an open-source platform for managing the ML and GenAI lifecycle. It records what
> you tried, versions what you decided to ship, and gives you a standard way to load and serve
> it. For LLM apps it adds tracing, prompt versioning and evaluation. It is a system of record.
> It does not orchestrate pipelines, store features or monitor drift by itself.

## How the pieces fit

| Concept | What it is | Where it appears here |
|---|---|---|
| Experiment | A named group of related runs | `01-tumor-classifier`, `02-support-bot` |
| Run | One execution: params, metrics, artifacts | Each candidate in the sweep |
| LoggedModel | First-class model entity in MLflow 3, with its own metrics | `models:/m-...` |
| Flavor / pyfunc | Framework-specific format plus a universal `predict()` interface | `mlflow.sklearn.log_model` |
| Signature | Input and output schema, enforced at serving | HTTP 400 in step 3 |
| Registered model | Named model with numbered versions | `tumor-classifier` v1, v2 |
| Alias | A movable pointer to a version | `@champion`, `@challenger` |
| Trace / span | Tree of steps for one request, with inputs, outputs, timing and tokens | Experiment `02-support-bot` |
| Prompt registry | Immutable prompt versions with aliases | `prompts:/support-answer@production` |
| Scorer / judge | Quality check written in code, or run by an LLM | `has_expected_answer` |

Storage has two halves, and interviewers often ask about this:

- The **backend store** is a SQL database that holds metadata: runs, params, metrics, registry.
- The **artifact store** is blob storage that holds files: models, plots.
- With `--serve-artifacts` (the default), the server proxies artifact traffic, so clients need
  no storage credentials.

## Before the interview

From `/root/MLFlow`:

```bash
./start.sh          # starts Docker Desktop, Ollama and MLflow in the background
./demo.sh all       # only if you want fresh data (about 2 minutes)
./stop.sh           # when you finish. Stops all three and keeps the demo data
```

Logs go to the `logs/` folder.

Open http://localhost:5000 in your Windows browser. Click through every screen once, because
the UI labels below are written from the data model and were not checked against a browser.

For a completely clean slate, run `./stop.sh`, then `./demo.sh reset`, then `./start.sh`.

## The demo script (about 10 minutes)

### 1. Tracking: "Every experiment is recorded"

Run `.venv/bin/python 01_train_and_track.py`, or show the existing data. Open experiment
`01-tumor-classifier` and expand the `model-sweep` parent run.

> "I ran six candidates as nested runs under one sweep. Each records its params, metrics, the
> exact dataset, a confusion matrix and the packaged model. Let me compare them."

Select the child runs and compare them. Results on the validation set:

| Candidate | F1 | ROC AUC |
|---|---|---|
| logreg C=0.001 | 0.893 | 0.989 |
| logreg C=0.1 | 0.979 | 0.997 |
| **logreg C=1.0** | **0.993** | **0.998** |
| forest 10 trees, depth 2 | 0.951 | 0.975 |
| forest 100 trees, depth 4 | 0.965 | 0.982 |
| forest 300 trees, no depth limit | 0.965 | 0.980 |

> "The simple scaled logistic regression beat the random forests, and that is exactly why you
> track baselines instead of assuming the complex model wins."

Point at this code in `01_train_and_track.py`: `start_run(nested=True)`, `log_input`,
`infer_signature`, and `log_metrics(..., model_id=info.model_id)`.

### 2. Registry: "Promotion is a gated decision"

Open Models, then `tumor-classifier`.

> "The data was split three ways. Train fits, validation chooses, and test is only for the
> promotion gate. I seeded a weak v1 as the incumbent. The best candidate became v2 with
> `@challenger`, scored 0.986 test F1 against 0.917, and so the script moved `@champion`."

> "Stages like Staging and Production are deprecated in favor of aliases. Consumers load
> `models:/tumor-classifier@champion`, so a release or rollback moves a pointer and needs no
> application code change."

If you run step 2 again, the new version ties the champion and is not promoted. That shows
the gate is real.

### 3. Serving: "Any model, same REST contract"

```bash
./demo.sh serve    # new terminal, wait for startup
./demo.sh score
```

Expected output: HTTP 200 with `{"predictions": [0, 1, 0]}`, matching the true labels, and then
HTTP 400 for a request with a missing column.

> "The serving layer does not care that this is scikit-learn, because it goes through pyfunc.
> The signature I logged at training time rejects a bad payload before it reaches the model."

### 4. Tracing: "See inside an LLM request"

Open experiment `02-support-bot`, then Traces. Open a trace from `04_trace_rag_app.py`:

```
request            CHAIN
  support_bot      AGENT
    retrieve       RETRIEVER
      Embeddings   EMBEDDING
    Completions    CHAT_MODEL
```

> "`mlflow.openai.autolog()` captures the LLM and embedding calls with one line. The
> `@mlflow.trace` decorators add my own steps. For a bad answer, I can see whether retrieval
> returned the wrong documents or the model ignored good ones. Those need different fixes."

Show the retrieved documents, token counts and latency. Filter by the session tag.

### 5. Prompts and evaluation: "Prompt changes need evidence"

Open the Prompts page, `support-answer`, and diff the versions. Then open the two evaluation runs.

| Prompt | Has expected answer | Concise |
|---|---|---|
| v1, loose | 78% | 89% |
| v2, grounded | 100% | 100% |

> "A prompt is a deployable artifact, so it gets versions and an alias like a model. I ran both
> versions over the same evaluation set. v2 won, so `@production` moved to it. The app loads
> the alias, so a prompt release needs no code deployment."

Open a failing v1 row to show answer-level detail. Say the limitation out loud: nine cases is a
demo, and I wrote v2 knowing what the test cases were.

### 6. Production shape: "This is how I would host it"

```bash
./demo.sh stack-up      # http://127.0.0.1:5050, then ./demo.sh stack-down
```

> "Locally I used SQLite. For a team, that becomes Postgres for concurrent writes, object
> storage for artifacts, and authentication in front. This Compose file shows the first step."

## Questions you are likely to get

**Why use aliases instead of stages?** Stages were a fixed list, and a version could only be in
one. Aliases are arbitrary named pointers, a version can have several, and tags carry extra state.

**What makes a run reproducible?** Code version, params, data reference, environment and seed.
MLflow records the git commit when run from a repo, logs pinned requirements with the model,
and tracks dataset metadata. It does not snapshot your data, so the source must be immutable or
versioned.

**After moving an alias, does the running server change?** Not automatically.
`mlflow models serve` resolves the alias when it starts and loads that model. You restart or
roll the deployment. The alias is the source of truth, and your deployment system acts on it.

**How would you deploy this at scale?** `mlflow models build-docker` makes an image for
Kubernetes or a managed endpoint. There is also a SageMaker integration. CI promotes the alias
after gates pass, then triggers a rolling or canary release.

**How do you secure it?** The server has no authentication by default. Use `--app-name basic-auth`
at minimum, or preferably SSO through a reverse proxy, plus TLS and private networking.
`--allowed-hosts` guards against bad Host headers. Unpickling a model can run code, so only load
models from trusted sources. MLflow 3.18 saves scikit-learn models with skops by default and made
this demo explicitly trust the tree type, which is why that argument appears in the code.

**Autolog or manual logging?** Autolog is great for coverage and speed. Add manual logging for
business metrics, custom plots and anything specific to the decision you need to make.

**What does MLflow not do?** Pipeline orchestration (Airflow, Kubeflow), feature stores (Feast),
data versioning (DVC, lakeFS), and classic drift monitoring (Evidently, or your observability stack).

**How does it compare to Weights & Biases?** W&B is a polished SaaS with strong collaboration
and visualization. MLflow is open source, self-hostable and has a broad model-packaging and
registry story. Databricks offers a managed MLflow. Compliance and hosting needs usually decide it.

**How does tracing work in production?** Export is asynchronous, so it adds little request
latency. Traces are OpenTelemetry-compatible. Use the lightweight `mlflow-tracing` package in
services, and sample high-volume traffic. Mask sensitive inputs before they are stored.

**Code scorers or LLM judges?** Use code for anything checkable, because it is cheap and
deterministic. Use judges for subjective qualities like groundedness or tone, and validate
judges against human labels. I tried a judge on `llama3.2:1b` here. It ran, but its reasoning was
incoherent, so I kept it out of the promotion gate. A judge needs a stronger model.

**What are the performance limits?** Logging many metrics one at a time is slow, so use
`log_batch` or asynchronous logging. Set retention rules, since deleted runs are soft-deleted
until `mlflow gc` removes them.

## Be straight about what this is

Describe it as "a demo I built to get hands-on with MLflow 3", and not as production experience.
You can say this confidently, because you ran it:

- The classic flow is real end to end, on a public dataset of 569 rows.
- Test F1 is used once per promotion here. Repeated gating on the same test set slowly leaks.
- The v1 seed was deliberately weak, so that there would be something to beat.
- The evaluation set has 9 cases, with simple substring checks.
- The Docker stack uses a local volume. The swap to real S3 was not tested here.
- Not demoed: AI Gateway, built-in judges in the gate, production monitoring, and user auth.

## If something breaks

| Symptom | Fix |
|---|---|
| Connection error on the LLM steps | Start `./demo.sh ollama`. Check the Windows firewall if it still fails. |
| `Connection refused` on port 5000 | Start `./demo.sh server`. |
| Port 5001 not answering | `./demo.sh serve` takes a few seconds, and needs the server up. |
| `stack-up` fails | Start Docker Desktop first. |
| Want a hosted model | Copy `.env.example` to `.env`, set `LLM_PROVIDER` and the key. |
