#!/usr/bin/env bash
# Reuse native Hermes dependencies read-only; never install into its environment.
set -eu
: "${HERMES_SOURCE:?Set HERMES_SOURCE to the inspected Hermes source directory}"
: "${HERMES_PYTHON:?Set HERMES_PYTHON to the matching Hermes Python interpreter}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# A relocated launcher may get Ariadne's source from the caller's Python path.
# Locate only our stdlib helper file; do not import packages to discover it.
HOST_SITE="$ROOT/src/hermes_ariadne/_host_site.py"
if [ ! -f "$HOST_SITE" ]; then
    IFS=: read -r -a SOURCE_PATHS <<< "${PYTHONPATH-}"
    for SOURCE_PATH in "${SOURCE_PATHS[@]}"; do
        if [ -n "$SOURCE_PATH" ] && [ -f "$SOURCE_PATH/hermes_ariadne/_host_site.py" ]; then
            HOST_SITE="$SOURCE_PATH/hermes_ariadne/_host_site.py"
            break
        fi
    done
fi
# Stdlib-only path discovery; disable all site/.pth startup and inherited paths.
SITE="$("$HERMES_PYTHON" -I -S "$HOST_SITE")"
export PYTHONPATH="$ROOT/src:$HERMES_SOURCE:$SITE${PYTHONPATH:+:$PYTHONPATH}"
# No sitecustomize/.pth startup imports before Python validates --hermes-home.
# Build binds HOME/HERMES_HOME in that fresh worker; rebuild needs no profile.
exec "$ROOT/.venv/bin/python" -S -m hermes_ariadne.compactor_demo "$@"
