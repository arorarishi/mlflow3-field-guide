"""Shared settings and data for the demo scripts."""
import os

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
SEED = 42

CLASSIC_EXPERIMENT = "01-tumor-classifier"
GENAI_EXPERIMENT = "02-support-bot"
REGISTERED_MODEL = "tumor-classifier"
PROMPT_NAME = "support-answer"

# Any OpenAI-compatible endpoint works, so switching provider is config, not code.
PROVIDERS = {
    "ollama": {
        "base_url": os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1"),
        "api_key": "ollama",
        "chat_model": "llama3.2:1b",
        "embed_model": "nomic-embed-text",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "api_key": os.getenv("OPENAI_API_KEY", ""),
        "chat_model": "gpt-4o-mini",
        "embed_model": "text-embedding-3-small",
    },
    "deepinfra": {
        "base_url": "https://api.deepinfra.com/v1/openai",
        "api_key": os.getenv("DEEPINFRA_API_KEY", ""),
        "chat_model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "embed_model": "BAAI/bge-base-en-v1.5",
    },
}
PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
LLM = PROVIDERS[PROVIDER]


def load_splits():
    """60/20/20 split: train to fit, validation to choose, test only to gate promotion."""
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    X = X.astype("float64")
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=0.25, stratify=y_rest, random_state=SEED
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def as_frame(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    return X.assign(target=y)
