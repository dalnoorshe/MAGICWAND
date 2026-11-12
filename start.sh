#!/usr/bin/env bash
# MAGICWAND — one command. Keeps the terminal attached so Ctrl-C stops it.
set -euo pipefail
cd "$(dirname "$0")"
exec python3 ./magicwand-launcher.py "$@"
