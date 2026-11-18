#!/usr/bin/env bash
# Starts Docker Desktop, Ollama and the MLflow server. Safe to run twice.
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p logs

DOCKER='/mnt/c/Program Files/Docker/Docker/resources/bin/docker.exe'
POWERSHELL=/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe
WINDOWS_IP="$(ip route | awk '/default/{print $3; exit}')"
OLLAMA_URL="http://${WINDOWS_IP}:11434/api/version"
MLFLOW_URL="http://127.0.0.1:5000/health"

docker_up() { "$DOCKER" version --format '{{.Server.Version}}' 2>/dev/null | tr -d '\r' | grep -Eq '^[0-9]'; }
http_up() { curl -sf -m 2 -o /dev/null "$1"; }

wait_for() { # wait_for <seconds> <command...>
  local deadline=$((SECONDS + $1))
  shift
  until "$@"; do
    [ "$SECONDS" -ge "$deadline" ] && return 1
    sleep 2
  done
}

report() { # report <name> <detail> <command...>
  local name=$1 detail=$2
  shift 2
  if "$@"; then
    echo "  OK    $name  $detail"
  else
    echo "  FAIL  $name  see logs/ in this folder"
    failed=1
  fi
}

if ! docker_up; then
  echo "Starting Docker Desktop..."
  "$POWERSHELL" -NoProfile -Command "Start-Process 'C:\Program Files\Docker\Docker\Docker Desktop.exe'"
fi

if ! http_up "$OLLAMA_URL"; then
  echo "Starting Ollama..."
  nohup setsid ./demo.sh ollama > logs/ollama.log 2>&1 < /dev/null &
fi

if ! http_up "$MLFLOW_URL"; then
  echo "Starting MLflow server..."
  nohup setsid ./demo.sh server > logs/mlflow.log 2>&1 < /dev/null &
fi

echo "Waiting for services..."
wait_for 180 docker_up
wait_for 60 http_up "$OLLAMA_URL"
wait_for 90 http_up "$MLFLOW_URL"

failed=0
report "Docker" "engine ready" docker_up
report "Ollama" "http://${WINDOWS_IP}:11434" http_up "$OLLAMA_URL"
report "MLflow" "http://localhost:5000" http_up "$MLFLOW_URL"
exit "$failed"
