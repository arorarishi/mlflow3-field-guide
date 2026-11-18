"""Shared helpers for every chapter of the MLflow book.

Each chapter imports this module, so settings live in one place:

* MLFLOW_TRACKING_URI  where runs go (default: the book server on port 5100)
* LLM_PROVIDER         fake | ollama | openai | deepinfra (default: fake)

The `fake` provider is a deterministic, offline stand-in for an LLM. It lets every GenAI
chapter run without a model or an API key. Switch to a real provider to see real LLM spans.
"""
from __future__ import annotations

import os

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")

import re
import subprocess
import sys
import urllib.request

import mlflow
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split

# Keep our prints in order with output from child processes.
sys.stdout.reconfigure(line_buffering=True)

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5100")
SEED = 42


def setup(experiment: str) -> mlflow.entities.Experiment:
    """Point MLflow at the book server and select (or create) the chapter's experiment."""
    mlflow.set_tracking_uri(TRACKING_URI)
    return mlflow.set_experiment(experiment)


def banner(title: str) -> None:
    print(f"\n=== {title} " + "=" * max(0, 70 - len(title)))


def last_trace():
    """The most recent trace from this process, after async export has finished."""
    trace_id = mlflow.get_last_active_trace_id()
    mlflow.flush_trace_async_logging()
    return mlflow.get_trace(trace_id)


def ui_link(path: str = "") -> str:
    return TRACKING_URI.rstrip("/") + "/#/" + path.lstrip("/")


# --------------------------------------------------------------------------- data


def classification_splits():
    """Breast cancer data: 60% train, 20% validation, 20% test."""
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    X = X.astype("float64")
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=0.25, stratify=y_rest, random_state=SEED
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def regression_splits():
    """Diabetes progression data: 80% train, 20% test."""
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    return train_test_split(X.astype("float64"), y, test_size=0.2, random_state=SEED)


def with_target(X: pd.DataFrame, y: pd.Series, name: str = "target") -> pd.DataFrame:
    return X.assign(**{name: y})


# --------------------------------------------------------------------------- LLM

PROVIDER = os.getenv("LLM_PROVIDER", "fake")

PROVIDERS = {
    "fake": {"base_url": None, "api_key": None, "chat_model": "fake-echo-1"},
    "ollama": {
        "base_url": os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1"),
        "api_key": "ollama",
        "chat_model": os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:1b"),
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "api_key": os.getenv("OPENAI_API_KEY", ""),
        "chat_model": "gpt-4o-mini",
    },
    "deepinfra": {
        "base_url": "https://api.deepinfra.com/v1/openai",
        "api_key": os.getenv("DEEPINFRA_API_KEY", ""),
        "chat_model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    },
}
LLM = PROVIDERS[PROVIDER]

KNOWLEDGE_BASE = {
    "refunds": "Acme refunds purchases within 30 days. Refunds reach the original payment method in 5 business days.",
    "shipping": "Standard shipping takes 4 days and costs $5. Orders above $50 ship free.",
    "warranty": "Acme hardware has a 2 year warranty. Accidental damage is not covered.",
    "password": "Reset a password from Settings > Security. Reset links expire after 15 minutes.",
    "plans": "The Pro plan costs $20 per month and includes 5 seats. Extra seats cost $4 each.",
}

_STOP = set(
    "a an the is are do does i my to of for in on how what when who much many long can you your "
    "it be with and acme".split()
)


def _words(text: str) -> set[str]:
    """Lower-cased content words with a crude plural strip, so 'refunds' matches 'refund'."""
    words = re.findall(r"[a-z0-9$]+", text.lower())
    return {w[:-1] if len(w) > 3 and w.endswith("s") else w for w in words if w not in _STOP}


def _fake_reply(messages: list[dict]) -> str:
    """A deterministic stand-in for an LLM that still reacts to the prompt.

    It answers with the context sentence that best overlaps the question. If nothing
    overlaps, it refuses when the prompt allows refusing ("I don't know") and otherwise
    answers confidently with an unrelated sentence, like a loosely prompted model would.
    """
    system = " ".join(m["content"] for m in messages if m["role"] == "system")
    question = [m["content"] for m in messages if m["role"] == "user"][-1]
    may_refuse = "i don't know" in system.lower()
    context = re.split(r"Context:|Background:", system)[-1]
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n", context) if s.strip()]
    scored = sorted(
        ((-len(_words(s) & _words(question)), i, s) for i, s in enumerate(sentences)),
    )
    if not scored or scored[0][0] == 0:
        return "I don't know." if may_refuse or not sentences else sentences[0]
    return scored[0][2]


@mlflow.trace(name="fake_chat_model", span_type="CHAT_MODEL")
def _fake_chat(messages: list[dict]) -> str:
    return _fake_reply(messages)


def llm_client():
    from openai import OpenAI

    return OpenAI(base_url=LLM["base_url"], api_key=LLM["api_key"], timeout=180)


def chat(messages: list[dict], temperature: float = 0.0) -> str:
    """One chat completion against the configured provider. Returns the reply text."""
    if PROVIDER == "fake":
        return _fake_chat(messages)
    response = llm_client().chat.completions.create(
        model=LLM["chat_model"], temperature=temperature, messages=messages
    )
    return response.choices[0].message.content


def judge_model() -> str | None:
    """Model URI for LLM judges, or None with the fake provider (judges are skipped).

    Judges call an OpenAI-compatible endpoint. Pointing OPENAI_BASE_URL at Ollama or
    DeepInfra lets the same `openai:/<model>` URI work for every provider.
    """
    if PROVIDER == "fake":
        return None
    os.environ["OPENAI_BASE_URL"] = LLM["base_url"]
    os.environ["OPENAI_API_KEY"] = LLM["api_key"] or os.environ.get("OPENAI_API_KEY", "")
    return f"openai:/{LLM['chat_model']}"


def require_real_llm(chapter: str) -> None:
    """Exit cleanly when a chapter needs a real model and none is configured."""
    if PROVIDER == "fake":
        print(f"[{chapter}] needs a real LLM. Set LLM_PROVIDER=ollama|openai|deepinfra. Skipping.")
        sys.exit(0)
    try:
        urllib.request.urlopen(LLM["base_url"].rstrip("/") + "/models", timeout=5)
    except Exception as error:  # noqa: BLE001 - any failure means "not reachable"
        if "401" not in str(error):
            print(f"[{chapter}] cannot reach {LLM['base_url']}: {error}. Skipping.")
            sys.exit(0)


# --------------------------------------------------------------------------- misc


def mlflow_cli() -> str:
    """Path to the `mlflow` executable that belongs to this interpreter."""
    candidate = os.path.join(os.path.dirname(sys.executable), "mlflow")
    return candidate if os.path.exists(candidate) else "mlflow"


def cli_env(**extra: str) -> dict:
    """Environment for child processes: this interpreter's bin dir first on PATH, so tools
    that MLflow shells out to (uvicorn, gunicorn) resolve even without an activated venv."""
    path = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")
    return {**os.environ, "PATH": path, "MLFLOW_TRACKING_URI": TRACKING_URI, **extra}


def run_cli(*args: str, check: bool = True, **kwargs) -> subprocess.CompletedProcess:
    print("$ mlflow " + " ".join(args))
    kwargs.setdefault("env", cli_env())
    return subprocess.run([mlflow_cli(), *args], check=check, text=True, **kwargs)
