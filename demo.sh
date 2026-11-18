#!/usr/bin/env bash
# Usage: ./demo.sh {ollama|server|classic|serve|score|genai|all|reset|stack-up|stack-down}
set -euo pipefail
cd "$(dirname "$0")"

[ -f .env ] && { set -a; source .env; set +a; }

# Under WSL NAT, Windows is reachable at the default gateway, and that IP can change on restart.
WINDOWS_IP="$(ip route | awk '/default/{print $3; exit}')"
export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://${WINDOWS_IP}:11434/v1}"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://127.0.0.1:5000}"
BIN=.venv/bin
export PATH="$PWD/$BIN:$PATH"
MODEL_ALIAS="models:/tumor-classifier@champion"
DOCKER='/mnt/c/Program Files/Docker/Docker/resources/bin/docker.exe'

case "${1:-}" in
  ollama)
    # Binds to the WSL adapter only, for this process only. No Windows setting is changed.
    OLLAMA_HOST="${WINDOWS_IP}:11434" WSLENV=OLLAMA_HOST \
      /mnt/c/Users/*/AppData/Local/Programs/Ollama/ollama.exe serve
    ;;
  server)
    exec $BIN/mlflow server --host 127.0.0.1 --port 5000 \
      --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts
    ;;
  classic)
    $BIN/python 01_train_and_track.py
    $BIN/python 02_register_and_promote.py
    ;;
  serve)
    exec $BIN/mlflow models serve -m "$MODEL_ALIAS" -p 5001 --env-manager local
    ;;
  score)
    $BIN/python 03_score_endpoint.py
    ;;
  genai)
    $BIN/python 04_trace_rag_app.py
    $BIN/python 05_prompts_and_eval.py
    ;;
  all)
    "$0" classic
    "$0" genai
    ;;
  reset)
    # Stop the server first. Removes only this demo's local state.
    rm -rf mlflow.db mlartifacts
    ;;
  stack-up)
    "$DOCKER" compose up -d --build
    echo "MLflow on Postgres: open http://127.0.0.1:5050 in a Windows browser"
    ;;
  stack-down)
    "$DOCKER" compose down
    ;;
  *)
    sed -n 2p "$0"
    exit 1
    ;;
esac
