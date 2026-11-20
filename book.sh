#!/usr/bin/env bash
# Runner for "Offline to Online", the MLflow 3 book.
#
#   ./book.sh start | stop | status | server   the book's tracking server (port 5100)
#   ./book.sh list                              chapters and their titles
#   ./book.sh run <NN> [<NN> ...]               run chapters, e.g. ./book.sh run 02 10
#   ./book.sh all                               run every chapter, print a summary
#   ./book.sh tour                              the 10-minute demo path (see WALKTHROUGH.md)
#   ./book.sh ollama                            WSL only: start Windows Ollama on the WSL gateway
#   ./book.sh reset                             delete the book's runs, models and traces
#   ./book.sh pdf                               build book/Offline-to-Online.pdf
#   ./book.sh docker-up | docker-down           Postgres + MLflow in Docker (port 5050)
#   ./book.sh docker-run <NN> [...]             run chapters inside Docker against that stack
#   ./book.sh docker-tour | docker-all          the tour, or every chapter, inside Docker
#
# Environment: LLM_PROVIDER (fake|ollama|openai|deepinfra), MLFLOW_TRACKING_URI, BOOK_PORT.
set -uo pipefail
cd "$(dirname "$0")"
ROOT=$PWD

[ -f .env ] && { set -a; source .env; set +a; }

BOOK_PORT=${BOOK_PORT:-5100}
DATA=$ROOT/book/.data
export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-http://127.0.0.1:$BOOK_PORT}
export LLM_PROVIDER=${LLM_PROVIDER:-fake}
export MLFLOW_DISABLE_AGENT_HINT=1
export PYTHONWARNINGS=${PYTHONWARNINGS:-ignore}

# Inside Docker, PYTHON=python is set. Locally, use the project virtualenv.
if [ -n "${PYTHON:-}" ]; then
  BIN_DIR=$(dirname "$(command -v "$PYTHON")")
elif [ -x .venv/bin/python ]; then
  PYTHON=$ROOT/.venv/bin/python
  BIN_DIR=$ROOT/.venv/bin
elif [[ "${1:-}" != docker-* ]]; then
  # The docker-* commands need only Docker: Python runs inside the containers.
  echo "No .venv found. Run ./setup.sh first, or use Docker: ./book.sh docker-tour"; exit 1
fi
[ -n "${BIN_DIR:-}" ] && export PATH="$BIN_DIR:$PATH"

# Under WSL, Ollama on Windows is reachable at the default gateway (see ./book.sh ollama).
if [ -z "${OLLAMA_BASE_URL:-}" ]; then
  if grep -qi microsoft /proc/version 2> /dev/null; then
    export OLLAMA_BASE_URL="http://$(ip route | awk '/default/{print $3; exit}'):11434/v1"
  else
    export OLLAMA_BASE_URL="http://127.0.0.1:11434/v1"
  fi
fi

if command -v docker > /dev/null && docker version > /dev/null 2>&1; then
  DOCKER=docker
else
  DOCKER='/mnt/c/Program Files/Docker/Docker/resources/bin/docker.exe'
fi

# The demo path from WALKTHROUGH.md: tune, register, gate, serve, trace, prompts, evaluate.
TOUR=(13 10 15 12 16 18 19)

server_up() { curl -sf -m 2 -o /dev/null "http://127.0.0.1:$BOOK_PORT/health"; }

chapter_dir() { # chapter_dir 7 -> book/07-models-flavors-signatures
  local n
  n=$(printf '%02d' "$((10#$1))" 2> /dev/null) || return 1
  compgen -G "book/$n-*" | head -1
}

run_chapter() {
  local dir
  dir=$(chapter_dir "$1") || { echo "no chapter '$1'"; return 2; }
  [ -n "$dir" ] || { echo "no chapter '$1'"; return 2; }
  echo
  echo "################ $(basename "$dir")  (LLM_PROVIDER=$LLM_PROVIDER)"
  "$PYTHON" "$dir/example.py"
}

require_server() {
  if [[ "$MLFLOW_TRACKING_URI" == "http://127.0.0.1:$BOOK_PORT" ]] && ! server_up; then
    echo "The book server is not running. Start it with: ./start.sh"; exit 1
  fi
}

case "${1:-}" in
  server)
    mkdir -p "$DATA"
    exec mlflow server --host 127.0.0.1 --port "$BOOK_PORT" \
      --backend-store-uri "sqlite:///$DATA/mlflow.db" --artifacts-destination "$DATA/artifacts"
    ;;
  start)
    if server_up; then echo "Book server already running: http://localhost:$BOOK_PORT"; exit 0; fi
    mkdir -p logs "$DATA"
    nohup setsid "$0" server > logs/book-server.log 2>&1 < /dev/null &
    for _ in $(seq 1 60); do server_up && break; sleep 1; done
    if server_up; then
      echo "Book server: http://localhost:$BOOK_PORT  (log: logs/book-server.log)"
    else
      echo "Server did not start. See logs/book-server.log"; exit 1
    fi
    ;;
  stop)
    pkill -f "mlflow server --host 127.0.0.1 --port $BOOK_PORT" && echo "Book server stopped" || echo "Book server was not running"
    ;;
  status)
    server_up && echo "UP   http://localhost:$BOOK_PORT" || echo "DOWN (./start.sh)"
    echo "LLM_PROVIDER=$LLM_PROVIDER  OLLAMA_BASE_URL=$OLLAMA_BASE_URL"
    ;;
  list)
    for dir in book/[0-9][0-9]-*/; do
      title=$(sed -n '1s/^# //p' "$dir/README.md" 2> /dev/null)
      printf '  %s\n' "${title:-$(basename "$dir")}"
    done
    ;;
  run)
    shift
    [ $# -gt 0 ] || { echo "usage: ./book.sh run <NN> [<NN> ...]"; exit 2; }
    require_server
    status=0
    for chapter in "$@"; do run_chapter "$chapter" || status=$?; done
    exit "$status"
    ;;
  all | tour)
    require_server
    mkdir -p logs
    passed=() failed=()
    if [ "$1" = tour ]; then
      chapters=("${TOUR[@]}")
    else
      chapters=()
      for dir in book/[0-9][0-9]-*/; do chapters+=("$(basename "$dir" | cut -c1-2)"); done
    fi
    for n in "${chapters[@]}"; do
      if run_chapter "$n" 2>&1 | tee "logs/chapter-$n.log"; [ "${PIPESTATUS[0]}" -eq 0 ]; then
        passed+=("$n")
      else
        failed+=("$n")
      fi
    done
    echo
    echo "Passed: ${#passed[@]}  ${passed[*]}"
    echo "Failed: ${#failed[@]}  ${failed[*]:-}"
    [ ${#failed[@]} -eq 0 ]
    ;;
  ollama)
    # Binds to the WSL adapter only, for this process only. No Windows setting is changed.
    gateway=$(ip route | awk '/default/{print $3; exit}')
    OLLAMA_HOST="$gateway:11434" WSLENV=OLLAMA_HOST \
      exec /mnt/c/Users/*/AppData/Local/Programs/Ollama/ollama.exe serve
    ;;
  pdf)
    "$PYTHON" book/build_pdf.py
    ;;
  reset)
    if server_up; then echo "Stop the server first: ./book.sh stop"; exit 1; fi
    rm -rf "$DATA" && echo "Removed $DATA"
    ;;
  docker-up)
    "$DOCKER" compose up -d --build postgres mlflow
    echo "MLflow (Postgres) at http://127.0.0.1:5050"
    ;;
  docker-down)
    "$DOCKER" compose --profile book down
    ;;
  docker-run)
    shift
    [ $# -gt 0 ] || { echo "usage: ./book.sh docker-run <NN> [<NN> ...]"; exit 2; }
    "$DOCKER" compose --profile book run --rm --build book run "$@"
    ;;
  docker-tour | docker-all)
    "$DOCKER" compose --profile book run --rm --build book "${1#docker-}"
    ;;
  *)
    sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
    ;;
esac
