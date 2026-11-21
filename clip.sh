#!/usr/bin/env bash
# macOS / Linux launcher. Setup there: python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
# plus ffmpeg and deno from your package manager (brew install ffmpeg deno).
DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONUTF8=1 PYTHONPATH="$DIR${PYTHONPATH:+:$PYTHONPATH}"
exec "$DIR/.venv/bin/python" -m clipper "$@"
