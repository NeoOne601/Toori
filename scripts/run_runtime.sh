#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -n "${TOORI_PYTHON:-}" ]]; then
  PYTHON_BIN="$TOORI_PYTHON"
elif command -v python3.11 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3.11)"
elif [[ -x "/Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11" ]]; then
  PYTHON_BIN="/Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11"
else
  echo "Toori runtime requires Python 3.11, but no python3.11 interpreter was found." >&2
  echo "Install Python 3.11 or set TOORI_PYTHON to a Python 3.11 binary." >&2
  exit 1
fi

TOORI_DATA_DIR="${TOORI_DATA_DIR:-.toori}"
HOST="${TOORI_HOST:-127.0.0.1}"
PORT="${TOORI_PORT:-7777}"

exec "$PYTHON_BIN" -m uvicorn cloud.api.main:app --host "$HOST" --port "$PORT" "$@"
