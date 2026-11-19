#!/usr/bin/env bash
# Start the MLflow tracking server.
#
#   ./start.sh            local server (SQLite)    http://localhost:5100   needs ./setup.sh
#   ./start.sh docker     Docker stack (Postgres)  http://127.0.0.1:5050   needs only Docker
#
# Stop with ./stop.sh. Your runs, models and traces are kept.
set -euo pipefail
cd "$(dirname "$0")"

case "${1:-local}" in
  local)  exec ./book.sh start ;;
  docker) exec ./book.sh docker-up ;;
  *)      sed -n '2,7p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
