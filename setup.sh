#!/usr/bin/env bash
# One-time setup: Python 3.12 virtualenv in .venv with everything the book and demo need.
# Safe to run again; it only installs what is missing or out of date.
#
#   ./setup.sh            create/update .venv
#   ./setup.sh --check    only verify the installation
set -euo pipefail
cd "$(dirname "$0")"

PY_VERSION=3.12
VENV=.venv
export MLFLOW_DISABLE_AGENT_HINT=1

ok()   { echo "  OK    $*"; }
fail() { echo "  FAIL  $*"; exit 1; }

verify() {
  "$VENV/bin/python" - <<'PY'
import importlib.util, sys
mods = ["mlflow", "sklearn", "pandas", "matplotlib", "skops", "openai", "optuna", "psutil", "flask_wtf", "reportlab"]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
if missing:
    sys.exit(f"missing packages: {missing}")
import mlflow
print(f"  OK    python {sys.version.split()[0]}, mlflow {mlflow.__version__}")
PY
}

if [ "${1:-}" = "--check" ]; then
  [ -x "$VENV/bin/python" ] || fail "no $VENV. Run ./setup.sh"
  verify
  exit 0
fi

echo "Setting up the MLflow book in $(pwd)"

if command -v uv > /dev/null; then
  ok "uv $(uv --version | awk '{print $2}')"
  [ -x "$VENV/bin/python" ] || uv venv --python "$PY_VERSION" "$VENV"
  uv pip install --python "$VENV/bin/python" -r requirements.txt
elif command -v "python$PY_VERSION" > /dev/null; then
  ok "python$PY_VERSION (uv not found, using venv + pip)"
  [ -x "$VENV/bin/python" ] || "python$PY_VERSION" -m venv "$VENV"
  "$VENV/bin/python" -m pip install --upgrade pip
  "$VENV/bin/python" -m pip install -r requirements.txt
else
  echo "Need uv or python$PY_VERSION. Install uv with:"
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

verify
chmod +x book.sh demo.sh start.sh stop.sh 2> /dev/null || true
[ -f .env ] || echo "  NOTE  no .env: the book uses the offline 'fake' LLM. See .env.example."

cat <<'MSG'

Next:
  ./book.sh start       start the book's tracking server (http://localhost:5100)
  ./book.sh list        list chapters
  ./book.sh run 02      run one chapter (or: ./book.sh all)
MSG
