#!/bin/sh
set -e

DATA_DIR="/app/data"
REVIEWS_DIR="/app/research reviews"

mkdir -p "$DATA_DIR" "$REVIEWS_DIR"

if [ ! -f "$DATA_DIR/papers.sqlite3" ]; then
  python main.py init-db
fi

exec "$@"
