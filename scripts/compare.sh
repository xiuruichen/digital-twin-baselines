#!/usr/bin/env bash
# usage: scripts/compare.sh <run-name> <gguf> <n-people>   (starts a server, runs, stops it)
cd "$(dirname "$0")/.."
powershell -NoProfile -File scripts/serve.ps1 "$2" > tools/server_$1.log 2>&1 &
until curl -s localhost:8080/health | grep -q ok; do sleep 3; done
.venv/Scripts/python.exe -u -m dteval run --name "$1" --n-people "$3" > "runs/$1.log" 2>&1
taskkill //F //IM llama-server.exe > /dev/null 2>&1
