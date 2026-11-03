#!/usr/bin/env bash
# Reuse native Hermes dependencies read-only; never install into its environment.
set -eu
: "${HERMES_SOURCE:?Set HERMES_SOURCE to the inspected Hermes source directory}"
: "${HERMES_PYTHON:?Set HERMES_PYTHON to the matching Hermes Python interpreter}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SITE="$("$HERMES_PYTHON" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
export PYTHONPATH="$ROOT/src:$HERMES_SOURCE:$SITE"
exec "$ROOT/.venv/bin/python" -m hermes_ariadne.compactor_demo "$@"
