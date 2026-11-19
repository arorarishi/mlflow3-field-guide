#!/usr/bin/env bash
# Stop everything this project started: the local server, any model server from
# chapters 12 or 21, and the Docker stack if it is up. Your data is kept.
# Docker Desktop and Ollama are left running.
set -uo pipefail
cd "$(dirname "$0")"

./book.sh stop

if pkill -f "mlflow models serve" 2> /dev/null; then
  echo "Model server stopped"
fi

if command -v docker > /dev/null && docker version > /dev/null 2>&1; then
  DOCKER=docker
else
  DOCKER='/mnt/c/Program Files/Docker/Docker/resources/bin/docker.exe'
fi
if "$DOCKER" version > /dev/null 2>&1 \
  && [ -n "$("$DOCKER" compose --profile book ps -q 2> /dev/null)" ]; then
  "$DOCKER" compose --profile book down && echo "Docker stack stopped"
fi
