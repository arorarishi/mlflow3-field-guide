#!/usr/bin/env bash
# Stops the MLflow server, Ollama and Docker Desktop. Demo data and volumes are kept.
# Stopping Docker Desktop stops every container on this machine, not only this demo's.
set -uo pipefail
cd "$(dirname "$0")"

DOCKER='/mnt/c/Program Files/Docker/Docker/resources/bin/docker.exe'
POWERSHELL=/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe
WSL=/mnt/c/Windows/System32/wsl.exe
VENV_PATTERN="$PWD/.venv/bin/"

docker_up() { "$DOCKER" version --format '{{.Server.Version}}' 2>/dev/null | tr -d '\r' | grep -Eq '^[0-9]'; }
windows_count() { "$POWERSHELL" -NoProfile -Command "@(Get-Process $1 -ErrorAction SilentlyContinue).Count" | tr -d '\r'; }

echo "Stopping MLflow..."
# Everything run from this project's virtualenv: server, workers and any model endpoint.
pkill -f "$VENV_PATTERN"
for _ in $(seq 1 10); do
  pgrep -f "$VENV_PATTERN" > /dev/null || break
  sleep 1
done
pkill -9 -f "$VENV_PATTERN"
sleep 1

echo "Stopping Ollama..."
"$POWERSHELL" -NoProfile -Command "Stop-Process -Name ollama,'ollama app' -Force -ErrorAction SilentlyContinue"

echo "Stopping Docker Desktop..."
if docker_up; then
  "$DOCKER" compose down 2>/dev/null
fi
timeout 60 "$DOCKER" desktop stop 2>/dev/null
# Fall back to ending the app if the clean shutdown did not finish.
if [ "$(windows_count "'Docker Desktop','com.docker.*'")" != "0" ]; then
  "$POWERSHELL" -NoProfile -Command "Get-Process 'Docker Desktop','com.docker.*' -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue"
  sleep 3
fi
# A leftover engine distro makes the next Docker Desktop start hang. Images and volumes persist.
"$WSL" --terminate docker-desktop > /dev/null 2>&1

failed=0
if pgrep -f "$VENV_PATTERN" > /dev/null || ss -ltn | grep -q ':5000 '; then
  echo "  FAIL  MLflow still running"; failed=1
else
  echo "  OK    MLflow stopped"
fi
if [ "$(windows_count ollama)" != "0" ]; then
  echo "  FAIL  Ollama still running"; failed=1
else
  echo "  OK    Ollama stopped"
fi
if [ "$(windows_count "'Docker Desktop','com.docker.*'")" != "0" ] \
  || "$WSL" --list --running 2>/dev/null | tr -d '\000\r' | grep -q '^docker-desktop'; then
  echo "  FAIL  Docker still running"; failed=1
else
  echo "  OK    Docker stopped"
fi
exit "$failed"
