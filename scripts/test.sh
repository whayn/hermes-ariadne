#!/usr/bin/env bash
set -eu
: "${HERMES_SOURCE:?Set HERMES_SOURCE to the inspected Hermes source directory}"
: "${HERMES_PYTHON:?Set HERMES_PYTHON to the Hermes Python interpreter}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SITE="$("$HERMES_PYTHON" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
# Reuse host dependencies read-only; install pytest/build/ruff in Ariadne's venv only.
export PYTHONPATH="$SITE"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
cd "$ROOT"
exec "$ROOT/.venv/bin/python" -m pytest "$@"
